from typing import Optional, Dict, Any
import PIL
import torch
import pytorch_lightning as pl
import wandb
from roma.roma_utils import compute_gt_warp
from roma.roma_losses import RoMaV2Loss, RoMaV1Loss
from training.lightning_module import ContrastiveLearningModule
# Assuming the RoMa V2 code is installed or in python path as 'romatch' or similar
# Replace this import with the actual path to the RoMaV2 class
try:
    from romav2 import RoMaV2
except ImportError:
    print("RoMaV2 not found")

try:
   from romatch import roma_indoor
except ImportError:
    print("RoMaV1 not found")

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import io
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

from roma.roma_viz_utils import visualize_debug, render_wandb_image
from roma.roma_utils import transform_params_to_identity

def analyze_channel_correlations(features):
    """
    features shape: [batch, num_patches, channels] = [2, 1600, 1024]
    """
    batch_size, num_patches, channels = features.shape
    
    # Reshape to combine batch and patches: [N, channels] where N = batch * num_patches
    N = batch_size * num_patches  # 2 * 1600 = 3200
    X = features.reshape(N, channels)
    
    # Center the data (remove mean from each channel)
    X_centered = X - X.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (N - 1)
    
    # Compute correlation matrix
    std_dev = torch.sqrt(torch.diag(cov_matrix))
    corr_matrix = cov_matrix / torch.outer(std_dev, std_dev)
    
    # Analyze correlation matrix
    # Exclude diagonal (self-correlation = 1)
    mask = ~torch.eye(channels, dtype=torch.bool)
    off_diag_correlations = corr_matrix[mask]
    
    # Statistics
    stats = {
        'mean_abs_correlation': off_diag_correlations.abs().mean().item(),
        'std_correlation': off_diag_correlations.std().item(),
        'max_correlation': off_diag_correlations.abs().max().item(),
        'median_abs_correlation': off_diag_correlations.abs().median().item(),
        'high_corr_fraction': (off_diag_correlations.abs() > 0.5).float().mean().item(),
        'very_high_corr_fraction': (off_diag_correlations.abs() > 0.7).float().mean().item()
    }
    
    return corr_matrix, stats

class LearnableUpsample(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class MMFEtoCNNandDinoV2Wrapper(nn.Module):
    def __init__(self, backbone_checkpoint_path, fine_from_vgg = True):
        super().__init__()
        self.backbone = ContrastiveLearningModule.load_from_checkpoint(backbone_checkpoint_path, 
                                                                        map_location="cpu", load_dino_weights=False,
                                                                        weights_only=False
                                                                        )
        self.features_adapter = nn.Sequential(
                                    nn.Conv2d(32, 256, 3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 512, 3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 1024, 1),
                                )
        self.fine_from_vgg = fine_from_vgg
        if self.fine_from_vgg:
            # Learnable upsamplers
            self.up_16_to_8 = LearnableUpsample(1024)
            self.up_8_to_4 = LearnableUpsample(1024)
            self.up_4_to_2 = LearnableUpsample(1024)
            self.up_2_to_1 = LearnableUpsample(1024)
    
    def forward(self, x, upsample=False):
        assert not upsample, "Upsample not supported for MMFE backbone"

        # x: [2B, C, H, W]
        if x.shape[-2:] != (256, 256):
            x = F.interpolate(x, (256, 256), mode="bilinear", align_corners=False)

        feats_16 = self.backbone(x)           # [2B, 32, 32, 32]
        feats_16 = self.features_adapter(feats_16)
        feats_16 = F.normalize(feats_16, dim=1)

        feats_8 = self.up_16_to_8(feats_16)    # 64×64
        feats_4 = self.up_8_to_4(feats_8)    # 64×64
        feats_2 = self.up_4_to_2(feats_4)    # 128×128
        feats_1 = self.up_2_to_1(feats_2)    # 256×256

        feats_8  = F.normalize(self.up_16_to_8(feats_16), dim=1) # 64×64
        feats_4  = F.normalize(self.up_8_to_4(feats_8), dim=1) # 128×128
        feats_2  = F.normalize(self.up_4_to_2(feats_4), dim=1) # 256×256
        feats_1  = F.normalize(self.up_2_to_1(feats_2), dim=1) # 512×512

        return {
            16: feats_16,
            8: feats_8,
            4: feats_4,
            2: feats_2,
            1: feats_1,
        }

class MMFEtoDinoV2Wrapper(nn.Module):
    def __init__(self, backbone_checkpoint_path, device="cpu"):
        super().__init__()
        self.device = device
        self.backbone = ContrastiveLearningModule.load_from_checkpoint(backbone_checkpoint_path, 
                                                                        map_location="cpu", load_dino_weights=False,
                                                                        weights_only=False
                                                                        )
        self.backbone_proj_dim = self.backbone.model_config.get("projection_dim", 32)
        self.roma_dino_dim = 1024
        if self.backbone_proj_dim != self.roma_dino_dim:
            self.features_adapter = nn.Sequential(
                                        nn.Conv2d(self.backbone_proj_dim, 256, 3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 512, 3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, self.roma_dino_dim, 1),
                                    )
        self.forward_features = self.forward
        self.out_norm = torch.nn.LayerNorm(1024, eps=1e-6) # for DINOv2
    
    def forward(self, im0):
        # Resize the images to 256x256
        if im0.shape[-2:] != (256, 256):
            im0 = F.interpolate(im0, size=(256, 256), mode="bilinear", align_corners=False)
        e0 = self.backbone(im0, normalize=False)
        e0 = F.interpolate(e0, size=(40, 40), mode="bilinear", align_corners=False)
        if self.backbone_proj_dim != self.roma_dino_dim:
            e0 = self.features_adapter(e0)
        B, C, H, W = e0.shape
        e0_flat = e0.flatten(2)  # Shape: B x 1024 x 1600
        e0_seq = e0_flat.transpose(1, 2)  # Shape: B x 1600 x 1024
        # e0_norm = F.normalize(e0_seq, p=2, dim=2) # Normalize again for 
        e0_norm = self.out_norm(e0_seq)

        # corr_matrix, stats = analyze_channel_correlations(e0_norm)

        # print("Correlation Statistics:")
        # for key, value in stats.items():
        #     print(f"{key}: {value:.4f}")

        return {
                'x_norm_patchtokens': e0_norm * 6.0 # Scale factor for DINOv2
            }

class DownSampledDinoV2Wrapper(nn.Module):
    def __init__(self, dinoV2_model, downsample_resolution=(256, 256), device="cpu"):
        super().__init__()
        self.device = device
        self.dinoV2_model = dinoV2_model
        self.forward_features = self.forward
        self.downsample_resolution = self.to_multiple_of_14(downsample_resolution)

    @staticmethod
    def to_multiple_of_14(resolution):
        """Convert resolution to nearest multiple of 14 (ceiling)."""
        def adjust(x):
            return ((x + 13) // 14) * 14
        
        if isinstance(resolution, int):
            return adjust(resolution)
        elif isinstance(resolution, tuple) and len(resolution) == 2:
            return (adjust(resolution[0]), adjust(resolution[1]))
        elif isinstance(resolution, list) and len(resolution) == 2:
            return [adjust(resolution[0]), adjust(resolution[1])]
        else:
            raise ValueError(f"Unsupported resolution format: {resolution}")
    
    def forward(self, im0):
        # Resize the images to the downsampled resolution
        im0 = F.interpolate(im0, size=self.downsample_resolution, mode="bilinear", align_corners=False)
        res = self.dinoV2_model.forward_features(im0)
        
        # Get patch tokens: [batch, num_patches, dim]
        patch_tokens = res['x_norm_patchtokens']  # [2, 361, 1024]
        
        # Reshape to spatial format: [batch, height, width, dim]
        batch_size = patch_tokens.shape[0]
        dim = patch_tokens.shape[2]
        spatial_tokens = patch_tokens.view(batch_size, 19, 19, dim)
        
        # Reorder dimensions for interpolation: [batch, dim, height, width]
        spatial_tokens = spatial_tokens.permute(0, 3, 1, 2)
        
        # Interpolate to 40x40
        resized_tokens = F.interpolate(
            spatial_tokens, 
            size=(40, 40), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Reshape back to flattened format: [batch, 1600, dim]
        resized_tokens = resized_tokens.permute(0, 2, 3, 1)
        flattened_tokens = resized_tokens.reshape(batch_size, 40*40, dim)

        # corr_matrix, stats = analyze_channel_correlations(resized_tokens)

        # print("Correlation Statistics:")
        # for key, value in stats.items():
        #     print(f"{key}: {value:.4f}")
        
        # Update the result dictionary
        res['x_norm_patchtokens'] = flattened_tokens
        
        return res

class RoMaFineTuner(pl.LightningModule):
    def __init__(self, matcher_name="romav1", lr=1e-4, weight_decay=1e-4, backbone_kwargs=None, 
                        mmfe_roma_checkpoint_path=None, use_pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = matcher_name.lower().strip()
        self.lr = lr
        self.weight_decay = weight_decay
        self.backbone_kwargs = backbone_kwargs
        self.use_pretrained = use_pretrained
        # Load model (constructed in eval mode by default)
        if matcher_name == "romav2":
            raise NotImplementedError("RoMaV2 is not supported yet")
            self.model = RoMaV2()
            self.model.bidirectional = False
            # Must force training mode because RoMa sets eval() internally
            self.model.train()
            self.model.requires_grad_(True)
            for param in self.model.parameters():
                param.requires_grad = True

            # Freeze only the DINO backbone
            self.model.f.requires_grad_(False)

            for param in self.model.f.parameters():
                param.requires_grad = False

            # Loss function
            self.loss_fn = RoMaV2Loss()
        elif matcher_name == "romav1":
            self.model = roma_indoor(device=self.device)

            if not self.use_pretrained:
                dino_state = self.extract_dino_state()
                self.reset_all_weights()
                self.restore_dino_state(dino_state)


            if self.backbone_kwargs.get("backbone_name") == "mmfe":
                assert self.backbone_kwargs.get("mmfe_checkpoint_path", None) is not None, "MMFE backbone checkpoint path is required"
                if self.backbone_kwargs.get("replace_dino_only", True):
                    mmfe_path = mmfe_roma_checkpoint_path if mmfe_roma_checkpoint_path is not None else self.backbone_kwargs.get("mmfe_checkpoint_path")
                    self.replace_roma_dino(mmfe_path)
                    # self.model.decoder.gps["16"].sigma_noise = 0.5
                else:
                    raise NotImplementedError("Replace the whole CNN+DINOv2 backbone is not supported yet")
                    self.replace_roma_backbone(self.backbone_kwargs.get("mmfe_checkpoint_path"))

            elif self.backbone_kwargs.get("use_downsampled_dino", None):
                assert self.backbone_kwargs.get("backbone_name") == "dinov2_vitb14", "Downsampled DINOv2 is only supported for DINOv2_vitb14 backbone"

                downsampled_dino_model = DownSampledDinoV2Wrapper(self.model.encoder.dinov2_vitl14[0], downsample_resolution=(256, 256))

                self.model.encoder.dinov2_vitl14[0] = downsampled_dino_model

            # Must force training mode because RoMa sets eval() internally
            self.model.train()
            self.model.requires_grad_(True)
            for param in self.model.parameters():
                param.requires_grad = True

            # Always freeze the DINO backbone
            if isinstance(self.model.encoder.dinov2_vitl14, list):
                # Dino is inside a list, so we need to access the first element
                # This is to hide the parameter (very ugly hack)
                self.model.encoder.dinov2_vitl14[0].requires_grad_(False)

                for param in self.model.encoder.dinov2_vitl14[0].parameters():
                    param.requires_grad = False

            else:
                # Dino is inside a list, so we need to access the first element
                # This is to hide the parameter (very ugly hack)
                self.model.encoder.dinov2_vitl14.requires_grad_(False)

                for param in self.model.encoder.dinov2_vitl14.parameters():
                    param.requires_grad = False

            if self.backbone_kwargs.get("freeze_dino_only", False):
                # Freeze only the DINO and fine vgg19 backbone
                self.model.encoder.requires_grad_(False)

                for param in self.model.encoder.parameters():
                    param.requires_grad = False

            # Loss function
            self.loss_fn = RoMaV1Loss()

    def reset_all_weights(self):
        def reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        for m in self.model.modules():
            reset(m)

    def extract_dino_state(self):
        dino = self.model.encoder.dinov2_vitl14
        if isinstance(dino, list):
            dino = dino[0]
        return {k: v.clone() for k, v in dino.state_dict().items()}

    def restore_dino_state(self, dino_state):
        dino = self.model.encoder.dinov2_vitl14
        if isinstance(dino, list):
            dino = dino[0]

        missing, unexpected = dino.load_state_dict(dino_state, strict=True)

        print("DINO restored")


    def configure_optimizers(self):
        # Filter only trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, 
                                      lr=self.hparams.lr, 
                                      weight_decay=self.hparams.weight_decay)
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
        return [optimizer], [scheduler]
    # ------------------------------
    # Forward wrappers
    # ------------------------------
    def evaluate_gp(self, im0: torch.Tensor, im1: torch.Tensor):
        """
        Evaluate Gaussian Process (GP) posterior with diagnostic metrics.
        Extracts features from input images and computes GP statistics.
        """
        # prefer explicit signature if available
        if isinstance(self.model.encoder.dinov2_vitl14, list):
            self.model.encoder.dinov2_vitl14[0] = self.model.encoder.dinov2_vitl14[0].to(im0.device)
            feats_0 = self.model.encoder.dinov2_vitl14[0].forward_features(im0)['x_norm_patchtokens']
            feats_1 = self.model.encoder.dinov2_vitl14[0].forward_features(im1)['x_norm_patchtokens']
        else:
            self.model.encoder.dinov2_vitl14 = self.model.encoder.dinov2_vitl14.to(im0.device)
            feats_0 = self.model.encoder.dinov2_vitl14.forward_features(im0)['x_norm_patchtokens']
            feats_1 = self.model.encoder.dinov2_vitl14.forward_features(im1)['x_norm_patchtokens']
        feats_0 = feats_0.view((feats_0.shape[0], int(np.sqrt(feats_0.shape[1])), int(np.sqrt(feats_0.shape[1])), feats_0.shape[2]))
        feats_1 = feats_1.view((feats_1.shape[0], int(np.sqrt(feats_1.shape[1])), int(np.sqrt(feats_1.shape[1])), feats_1.shape[2]))
        feats_0 = feats_0.detach().to("cpu").permute(0, 3, 1,2)
        feats_1 = feats_1.detach().to("cpu").permute(0, 3, 1,2)
        gp = self.model.decoder.gps["16"].to("cpu")
        # Test with only the first 10 tokens
        with torch.no_grad():
            test_0 = feats_0[:, :, :10, :10] 
            test_1 = feats_1[:, :, :10, :10]
            print(f"Testing with shape: {test_0.shape}") # Should be [1, 1024, 10, 10]
            posterior = gp(test_0, test_1)
            # posterior: [B, C, H, W]
            T = 0.07
            mu = posterior.flatten(2).transpose(1, 2)  # [B, HW, C]
            mu = F.normalize(mu, dim=-1)

            sim = torch.matmul(mu, mu.transpose(1, 2))  # [B, HW, HW]
            p = F.softmax(sim / T, dim=-1)
            entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)

            print(f"GP Min: {posterior.min().item()}, Max: {posterior.max().item()}")
            print(f"GP Mean: {posterior.mean().item()}, GP Std: {posterior.std().item()}")

            print(f"GP Entropy: {entropy.mean().item()} | GP Entropy Std: {entropy.std().item()}")


            phi_A = test_0.flatten(2).transpose(1, 2)  # [B, HW, C]
            phi_A = F.normalize(phi_A, dim=-1)
            phi_B = test_1.flatten(2).transpose(1, 2)  # [B, HW, C]
            phi_B = F.normalize(phi_B, dim=-1)

            K = torch.matmul(phi_B, phi_B.transpose(1, 2))
            eigvals = torch.linalg.eigvalsh(K[0])

            print(f"GP Eigenvalues Min: {eigvals.min().item()}, Max: {eigvals.max().item()}")

            sim = torch.matmul(phi_A, phi_B.transpose(1, 2))
            top1 = sim.max(dim=-1).values
            top10 = sim.topk(10, dim=-1).values.mean(dim=-1)

            gap = top1 - top10

            print(f"Gap Min: {gap.mean().item()}, Std: {gap.std().item()}")

            dx = posterior[:, :, 1:, :] - posterior[:, :, :-1, :]
            dy = posterior[:, :, :, 1:] - posterior[:, :, :, :-1]

            print(f"dx Norm: {dx.norm().item()}, dy Norm: {dy.norm().item()}")

            print("Success!")

    def forward(self, im0: torch.Tensor, im1: torch.Tensor, batch_dict: Optional[Dict[str, Any]] = None):
        """
        Unified forward wrapper.
        - For RoMaV2 you'd call model(im0, im1)
        - For RoMaV1 (roma_indoor) you must build a 'batch' dict containing features used by the model decoder.
          The roma_indoor.forward in your snippet takes `batch` and returns corresps dict.
        We detect model type and call appropriately.
        """
        if self.model_name == "romav1":
            if batch_dict is None:
                batch = {"im_A": im0, "im_B": im1}
            with torch.cuda.amp.autocast(enabled=False):
                try:
                    # self.evaluate_gp(im0, im1)
                    return self.model.forward(batch, batched=True, upsample=False, scale_factor=1)

                except TypeError:
                    # try positional style
                    return self.model(batch, batched=True, upsample=False, scale_factor=1)
        else:
            # romav2 or other - would be model(im0, im1)
            return self.model(im0, im1)

    def match(self, im0: PIL.Image.Image, im1: PIL.Image.Image):
        return self.model.match(im0, im1)

    def sample(self, preds, certainty, num_samples=5000):
        return self.model.sample(preds, certainty, num_samples)

    def to_pixel_coordinates(self, matches, H_A, W_A, H_B=None, W_B=None):
        return self.model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    def training_step(self, batch, batch_idx):
        
        # 1. Forward Pass
        out = self.forward(batch['modality_0'], batch['modality_1_noise'])
        
        # 2. Compute Ground Truth Warp
        # We need the warp that maps modality_0_noise -> modality_1_noise
        B, C, H, W = batch['modality_0'].shape
        
        common_params = transform_params_to_identity(batch['transform_params'])
        noise_params = batch['noise_params']
        with torch.no_grad():
            gt_warp, gt_mask = compute_gt_warp(
                common_params,
                noise_params,
                H, W, self.device
            )
        # -------------------------------------------------------------

        # 3. Compute Loss
        loss, log_metrics = self.loss_fn(out, gt_warp, gt_mask)
        
        # 4. Log
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(log_metrics)
        
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch['modality_0'], batch['modality_1_noise'])
        
        # Calculate GT (Same logic as train)
        params0 = batch['transform_params']

        params1 = batch['noise_params']
        B, C, H, W = batch['modality_0'].shape

        params0 = transform_params_to_identity(params0)
            
        gt_warp, gt_mask = compute_gt_warp(params0, params1, H, W, self.device)

        # visualize_debug(batch, gt_warp, gt_mask, "val_step")
        
        loss, metrics = self.loss_fn(out, gt_warp, gt_mask)
        self.log('val_loss', loss, prog_bar=True)
        
        # 4. Visualization (Only for first batch)
        if batch_idx == 0:
            # Get prediction [B, H, W, 2]
            if self.model_name == "romav2":
                pred_warp = out['warp_AB']
            else:
                pred_warp = out[1]["flow"].permute(0, 2, 3, 1)

            # A) Render Correspondence Plot (The "Proof" of alignment)
            vis_plot = render_wandb_image(batch, pred_warp, gt_warp, gt_mask)
            
            # B) Render Warp Field as RGB (To check smoothness)
            # Normalize [-1, 1] -> [0, 1] for RGB visualization
            # R channel = X coord, G channel = Y coord, B channel = 0
            warp_vis = pred_warp[0].detach().cpu() # [H, W, 2]
            warp_vis_rgb = torch.zeros(H, W, 3)
            warp_vis_rgb[..., 0] = (warp_vis[..., 0] + 1) / 2.0
            warp_vis_rgb[..., 1] = (warp_vis[..., 1] + 1) / 2.0
            
            # Log to W&B
            try:
                self.logger.experiment.log({
                    "val_match_plot": [wandb.Image(vis_plot, caption="Predicted Matches")],
                    "val_warp_field": [wandb.Image(warp_vis_rgb.numpy(), caption="Predicted Warp (RG=XY)")]
                })
            except AttributeError:
                # Handle cases where logger might not be attached during sanity checks
                pass


    def replace_roma_backbone(self, backbone_checkpoint_path, backbone_model=None):
        """
        Replace the RoMa backbone with a new one.
        Input:
        - backbone_checkpoint_path: path to the new backbone checkpoint
        Output:
        - None
        """
        if backbone_model is None:
            backbone_model = MMFEtoCNNandDinoV2Wrapper(backbone_checkpoint_path)
        self.model.encoder = backbone_model
    
    def replace_roma_dino(self, backbone_checkpoint_path, backbone_model=None):
        """
        Replace the RoMa DINOv2 backbone with a new one.
        Input:
        - backbone_checkpoint_path: path to the new backbone checkpoint
        Output:
        - None
        """
        if backbone_model is None:
            backbone_model = MMFEtoDinoV2Wrapper(backbone_checkpoint_path)
        self.model.encoder.dinov2_vitl14 = backbone_model