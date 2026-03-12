#!/usr/bin/env python3
"""
Validation script (Hydra-based)

This script:
1) Builds a UnifiedDataset for validation from Hydra config
2) Iterates over the dataset and computes validation metrics (loss + retrieval accuracy)
3) Aggregates metrics across batches (average)
4) Stores results to a pandas DataFrame, prints it, and saves it to disk

Run:
  python scripts/validate.py
  # overrides example:
  python scripts/validate.py \
    model.checkpoint=/abs/path/to/ckpt.ckpt \
    data.cubicasa.path=/abs/cubicasa5k data.cubicasa.val=/abs/cubicasa5k/val.txt \
    data.structured3d.path=/abs/Structured3D data.structured3d.val=/abs/Structured3D/val.json \
    data.image_size='[256,256]' val.batch_size=8 val.num_workers=8 \
    logging.output_csv=/abs/outputs/validation_metrics.csv
"""

import os
import sys
from typing import Dict, Any, List

import cv2
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from tqdm import tqdm

from dataloading.inversible_tf import warp_feature_map_batch
from dataloading.unified_dataset import UnifiedDataset
from dataloading.dual_transforms import PairRandomAffine, PairToPIL, PairResize, PairGrayscale, PairToTensor, PairNormalize
from inference.tta import vote_for_best_augmentation
from training.lightning_module import ContrastiveLearningModule, load_contrastive_model_from_checkpoint

from torch.utils._pytree import tree_map

from mmfe_utils.tensor_utils import torch_erode


def move_to_device(batch, device):
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if torch.is_tensor(x) else x,
        batch
    )


def visualize_debug_images(modality_0, modality_1_noise, warp_mask_1=None, save_path="debug_viz.png"):
    """
    Visualize modality_0, modality_1_noise, and warp_mask_1 for the first example in the batch.
    
    Args:
        modality_0: Tensor of shape [B, C, H, W] - first modality
        modality_1_noise: Tensor of shape [B, C, H, W] - second modality with noise
        warp_mask_1: Optional tensor of shape [B, H, W] - warp mask
        save_path: Path to save the visualization
    """
    # Take first example from batch
    img0 = modality_0[0].cpu()
    img1_noise = modality_1_noise[0].cpu()
    
    # Denormalize images (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img0_denorm = img0 * std + mean
    img1_noise_denorm = img1_noise * std + mean
    
    # Clamp to valid range [0, 1]
    img0_denorm = torch.clamp(img0_denorm, 0, 1)
    img1_noise_denorm = torch.clamp(img1_noise_denorm, 0, 1)
    
    # Convert to numpy and transpose for matplotlib (H, W, C)
    img0_np = img0_denorm.permute(1, 2, 0).numpy()
    img1_noise_np = img1_noise_denorm.permute(1, 2, 0).numpy()
    
    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot modality_0
    axes[0].imshow(img0_np)
    axes[0].set_title('Modality 0 (Original)')
    axes[0].axis('off')
    
    # Plot modality_1_noise
    axes[1].imshow(img1_noise_np)
    axes[1].set_title('Modality 1 (Noisy)')
    axes[1].axis('off')
    
    # Plot warp_mask_1 if available
    if warp_mask_1 is not None:
        mask_np = warp_mask_1[0].permute(1, 2, 0).cpu().numpy()
        im = axes[2].imshow(mask_np, cmap='gray')
        axes[2].set_title('Warp Mask 1')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, 'No warp mask\n(angle.sum() <= 1)', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Warp Mask 1')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug visualization saved to: {save_path}")


def create_val_dataset(cfg: DictConfig) -> UnifiedDataset:
    # Transforms similar to train_contrastive.py (validation path)
    mean = torch.tensor(list(cfg.transforms.mean))
    std = torch.tensor(list(cfg.transforms.std))

    if cfg.transforms.tf_difficulty == "easy":
        transforms = {
            "degrees": 0,
            "translate": [0.0, 0.0],
            "scale": [1.0, 1.0],
        }
    elif cfg.transforms.tf_difficulty == "medium":
        transforms = {
            "degrees": 10,
            "translate": [0.1, 0.1],
            "scale": [0.8, 1.2],
        }
    elif cfg.transforms.tf_difficulty == "hard":
        transforms = {
            "degrees": 180,
            "translate": [0.2, 0.2],
            "scale": [0.6, 1.4],
        }
    elif cfg.transforms.tf_difficulty == "rot_only":
        transforms = {
            "degrees": 180,
            "translate": [0.0, 0.0],
            "scale": [1.0, 1.0],
        }

    dual_transform_val = [
        PairToPIL(),
        PairResize(tuple(cfg.data.image_size)),
        PairGrayscale(num_output_channels=3),
        PairToTensor(),
        PairRandomAffine(degrees=180, translate=[0.0, 0.0], scale=[1.0, 1.0]),
        PairNormalize(mean=mean, std=std),
    ]

    filler = (1-mean)/std
    noise_transform_val = [
        PairRandomAffine(degrees=transforms["degrees"], translate=transforms["translate"], scale=transforms["scale"], filler=filler),
    ]


    val_configs = [
        {
            "type": "cubicasa5k",
            "args": {
                "root_dir": to_absolute_path(cfg.data.cubicasa.path),
                "sample_ids_file": to_absolute_path(cfg.data.cubicasa.val),
                "image_size": tuple(cfg.data.image_size),
            },
        },
        {
            "type": "structured3d",
            "args": {
                "root_dir": to_absolute_path(cfg.data.structured3d.path),
                "scene_ids_file": to_absolute_path(cfg.data.structured3d.val),
                "image_size": tuple(cfg.data.image_size),
            },
        },
    ]

    dataset = UnifiedDataset(dataset_configs=val_configs, common_transform=dual_transform_val, invertible_transform=noise_transform_val)
    return dataset


def compute_batch_metrics(model, batch: Dict[str, Any], cfg: DictConfig = None, batch_idx: int = 0) -> Dict[str, float]:
    modality_0 = batch["modality_0"].to(next(model.parameters()).device)
    modality_1 = batch["modality_1"].to(next(model.parameters()).device) if not cfg.use_m0_as_m1 else batch["modality_0"].to(next(model.parameters()).device)
    modality_1_noise = batch["modality_1_noise"].to(next(model.parameters()).device) if not cfg.use_m0_as_m1 else batch["modality_0_noise"].to(next(model.parameters()).device)
    embeddings_0 = model.get_embeddings(modality_0)
    # # TODO: Remove this
    if cfg.transforms.hardcoded_rot != None:
        harcoded_rot = cfg.transforms.hardcoded_rot
        filler = (1-torch.tensor(cfg.transforms.mean))/torch.tensor(cfg.transforms.std)
        filler = filler.view(-1, 1, 1).to(modality_1_noise.device).unsqueeze(0)
        # mask = torch.ones(modality_1_noise.shape[0], 1, modality_1_noise.shape[2], modality_1_noise.shape[3], device=modality_1_noise.device)
        mask = batch["transform_params"]["valid_mask"]
        mask = TF.affine(mask, harcoded_rot, [0., 0.], 1.0, 0.0, interpolation=TF.InterpolationMode.NEAREST)
        mask = torch_erode(mask, kernel_size=3, iterations=1)
        modality_1_noise = TF.affine(modality_1, harcoded_rot, [0., 0.], 1.0, 0.0, interpolation=TF.InterpolationMode.BILINEAR)
        modality_1_noise = torch.where(~mask.repeat(1, modality_1_noise.shape[1], 1, 1).bool(), filler, modality_1_noise)
        batch["modality_1_noise"] = modality_1_noise
        batch["noise_params"] = {
            "angle": torch.tensor([harcoded_rot]* batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device),
            "translate": [torch.zeros(batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device),
                                            torch.zeros(batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device)],
            "scale": torch.tensor([1.0]* batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device),
            "shear": torch.tensor([0.0]* batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device),
            "image_size": batch["noise_params"]["image_size"],
            "valid_mask": mask,
        }

    if cfg is not None and getattr(cfg, 'viz_debug', False):
        # Denormalize images using the same normalization values as used in training
        mean = torch.tensor(cfg.transforms.mean).view(3, 1, 1)
        std = torch.tensor(cfg.transforms.std).view(3, 1, 1)
        
        modality_0_denorm = modality_0[0].cpu() * std + mean
        modality_1_denorm = modality_1[0].cpu() * std + mean
        modality_1_noise_denorm = modality_1_noise[0].cpu() * std + mean
        
        # Clamp to valid range [0, 1] and convert to numpy
        modality_0_denorm = torch.clamp(modality_0_denorm, 0, 1)
        modality_1_denorm = torch.clamp(modality_1_denorm, 0, 1)
        modality_1_noise_denorm = torch.clamp(modality_1_noise_denorm, 0, 1)
        
        modality_0_show = modality_0_denorm.permute(1, 2, 0).numpy()
        modality_1_show = modality_1_denorm.permute(1, 2, 0).numpy()
        modality_1_noise_show = modality_1_noise_denorm.permute(1, 2, 0).numpy()

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(modality_0_show)
        axs[0].set_title("modality_0")
        axs[0].axis('off')
        axs[1].imshow(modality_1_show)
        axs[1].set_title("modality_1")
        axs[1].axis('off')
        axs[2].imshow(modality_1_noise_show)
        axs[2].set_title("modality_1_noise")
        axs[2].axis('off')
        plt.show()

    with torch.no_grad():
        if cfg.tta != 1 and cfg.mode in ["gt_tta", "tta"]:
            aug_step = 360 / cfg.tta
            aug_embeddings = []
            all_aug_params = []
            filler = (1-torch.tensor(cfg.transforms.mean))/torch.tensor(cfg.transforms.std)
            filler = filler.view(-1, 1, 1).to(modality_1_noise.device).unsqueeze(0)
            embeddings_1_noisy_aug = model.get_embeddings(modality_1_noise)
            aug_embeddings.append(embeddings_1_noisy_aug)
            all_aug_params.append({"angle": 0, "translate": [0., 0.], "scale": 1.0, 
                                    "shear": 0.0, "image_size": modality_1_noise.shape[-2:]})

            for i in range(1, cfg.tta):
                # Rotate modality_1_noise with angle i*20 degrees
                angle = i * aug_step
                print(f"Augmentation {i} of {cfg.tta}: Angle {angle}")
                mask = batch["noise_params"]["valid_mask"]
                mask = TF.affine(mask, angle, [0., 0.], 1.0, 0.0, interpolation=TF.InterpolationMode.NEAREST)
                mask = torch_erode(mask, kernel_size=3, iterations=1)

                modality_1_noise_rot = TF.affine(modality_1_noise, angle, [0., 0.], 1.0, 0.0, interpolation=TF.InterpolationMode.BILINEAR)
                modality_1_noise_rot = torch.where(~mask.repeat(1, modality_1_noise_rot.shape[1], 1, 1).bool(), filler, modality_1_noise_rot)
                embeddings_1_noisy_aug = model.get_embeddings(modality_1_noise_rot)

                # modality_1_noise_rot_show = (modality_1_noise_rot[0] / modality_1_noise_rot[0].max() * 255).permute(1, 2, 0).cpu().numpy()
                # modality_1_noise_show = (modality_1_noise[0] / modality_1_noise[0].max() * 255).permute(1, 2, 0).cpu().numpy()
                # modality_1_noise_rot_show = modality_1_noise_rot_show.astype(np.uint8)
                # modality_1_noise_show = modality_1_noise_show.astype(np.uint8)
                # cv2.imshow("modality_1_noise_rot", modality_1_noise_rot_show)
                # cv2.imshow("modality_1_noise", modality_1_noise_show)
                # cv2.waitKey(0)

                # Create parameters for this augmentation
                params_aug = {}
                params_aug["angle"] = angle
                params_aug["translate"] = [0., 0.]
                params_aug["scale"] = 1.0
                params_aug["shear"] = 0.0
                params_aug["image_size"] = modality_1_noise.shape[-2:]

                aug_embeddings.append(embeddings_1_noisy_aug)
                all_aug_params.append(params_aug)

            # Vote for the best augmentation
            gt_rot = batch["noise_params"]["angle"]
            gt_aug = ((cfg.tta+(360-gt_rot+aug_step/2)//aug_step)%cfg.tta).to(torch.int32)

            aug_embeddings = torch.stack(aug_embeddings)  # (F, B, C, H, W)
            best_embedding_1_aug, best_aug_idx, votes_per_aug = vote_for_best_augmentation(embeddings_0, aug_embeddings)
            warp_mask_1 = None

            if cfg.mode == "gt_tta":
                best_aug_idx = gt_aug
                b_idx = torch.arange(modality_1_noise.shape[0], device=aug_embeddings.device)
                best_embedding_1_aug = aug_embeddings[best_aug_idx, b_idx]


            accuracy_tta = ((best_aug_idx - gt_aug)==0).sum()/len(best_aug_idx)

            print(f"TTA Accuracy: {accuracy_tta}")


            selected_params = {
                "angle": [],
                "translate": [],
                "scale": [],
                "shear": [],
                "image_size": [],
                "valid_mask": [],
            }
            for i in range(len(best_aug_idx)):
                best_idx = best_aug_idx[i]
                selected_params["angle"].append(all_aug_params[best_idx]["angle"])
                selected_params["translate"].append(all_aug_params[best_idx]["translate"])
                selected_params["scale"].append(all_aug_params[best_idx]["scale"])
                selected_params["shear"].append(all_aug_params[best_idx]["shear"])
                selected_params["image_size"].append(all_aug_params[best_idx]["image_size"])

                valid_mask = TF.affine(batch["noise_params"]["valid_mask"][i], 
                                        all_aug_params[best_idx]["angle"], all_aug_params[best_idx]["translate"], 
                                        all_aug_params[best_idx]["scale"], all_aug_params[best_idx]["shear"], 
                                        interpolation=TF.InterpolationMode.NEAREST)
                valid_mask = torch_erode(valid_mask, kernel_size=3, iterations=1)
                selected_params["valid_mask"].append(valid_mask)    
                
            selected_params["angle"] = torch.tensor(selected_params["angle"], device=modality_1_noise.device)
            selected_params["translate"] = torch.tensor(selected_params["translate"], device=modality_1_noise.device)
            selected_params["scale"] = torch.tensor(selected_params["scale"], device=modality_1_noise.device)
            selected_params["shear"] = torch.tensor(selected_params["shear"], device=modality_1_noise.device)
            selected_params["image_size"] = selected_params["image_size"][0]
            selected_params["valid_mask"] = torch.stack(selected_params["valid_mask"])
            
            wrapped_embeddings_1_aug, warp_mask_1_aug = warp_feature_map_batch(
                            best_embedding_1_aug, selected_params, image_size=selected_params["image_size"], 
                            align_corners=False, return_mask=True, og_valid_mask=selected_params["valid_mask"]
                        )

            wrapped_embeddings_1, warp_mask_1 = warp_feature_map_batch(
                            wrapped_embeddings_1_aug, batch["noise_params"], image_size=batch["noise_params"]["image_size"], 
                            align_corners=False, return_mask=True, og_valid_mask=warp_mask_1_aug
                        )

            # wrapped_embeddings_1, warp_mask_1 = warp_feature_map_batch(
            #     best_embedding_1_aug, [selected_params, batch["noise_params"]], image_size=batch["noise_params"]["image_size"], 
            #     align_corners=False, return_mask=True, og_valid_mask=selected_params["valid_mask"]
            # )

            # all_wrapped_embeddings_1 = []
            # all_warp_mask_1 = []
            # for i in range(len(best_embedding_1_aug)):
            #     wrapped_embeddings_1 = TF.affine(best_embedding_1_aug[i], -harcoded_rot-selected_params["angle"][i].item(), 
            #                                 selected_params["translate"][i].tolist(), 
            #                                 selected_params["scale"][i].item(), 
            #                                 selected_params["shear"][i].item(), interpolation=TF.InterpolationMode.BILINEAR)
            #     # wrapped_embeddings_1 = TF.affine(wrapped_embeddings_1, -batch["noise_params"]["angle"][i].item(), 
            #     #                             [batch["noise_params"]["translate"][0][i].item(), batch["noise_params"]["translate"][1][i].item()], 
            #     #                             batch["noise_params"]["scale"][i].item(), 
            #     #                             batch["noise_params"]["shear"][i].item(), interpolation=TF.InterpolationMode.BILINEAR)
            #     all_wrapped_embeddings_1.append(wrapped_embeddings_1)
            #     warp_mask_1 = TF.affine(selected_params["valid_mask"][i], -harcoded_rot-selected_params["angle"][i].item(), 
            #                                 selected_params["translate"][i].tolist(), 
            #                                 selected_params["scale"][i].item(), 
            #                                 selected_params["shear"][i].item(), interpolation=TF.InterpolationMode.NEAREST)
            #     # warp_mask_1 = TF.affine(warp_mask_1, -batch["noise_params"]["angle"][i].item(), 
            #     #                             [batch["noise_params"]["translate"][0][i].item(), batch["noise_params"]["translate"][1][i].item()], 
            #     #                             batch["noise_params"]["scale"][i].item(), 
            #     #                             batch["noise_params"]["shear"][i].item(), interpolation=TF.InterpolationMode.NEAREST)
            #     all_warp_mask_1.append(warp_mask_1)

            # wrapped_embeddings_1 = torch.stack(all_wrapped_embeddings_1)
            # warp_mask_1 = torch.stack(all_warp_mask_1)

            warp_mask_1 = torch_erode(warp_mask_1, kernel_size=3, iterations=1)
            warp_mask_1 = TF.resize(warp_mask_1, size=wrapped_embeddings_1.shape[-2:], interpolation=TF.InterpolationMode.NEAREST)

        elif cfg.mode == "pre":
            # Pre-Affine
            mask_img_tf = batch["transform_params"]["valid_mask"]
            if cfg.transforms.hardcoded_rot != None:
                mask_img = TF.affine(mask_img_tf, harcoded_rot, [0., 0.], 1.0, 0.0, interpolation=TF.InterpolationMode.NEAREST)
                mask_img = torch_erode(mask_img, kernel_size=3, iterations=1)
            else:
                all_masks = []
                for i in range(len(batch["noise_params"]["angle"])):
                    mask_img = TF.affine(mask_img_tf[i], batch["noise_params"]["angle"][i].item(), 
                                            [batch["noise_params"]["translate"][0][i].item(), batch["noise_params"]["translate"][1][i].item()], 
                                            batch["noise_params"]["scale"][i].item(), 
                                            batch["noise_params"]["shear"][i].item(),
                                            interpolation=TF.InterpolationMode.NEAREST)
                    mask_img = torch_erode(mask_img, kernel_size=3, iterations=1)
                    all_masks.append(mask_img)
                mask_img = torch.stack(all_masks)

            filler = (1-torch.tensor(cfg.transforms.mean))/torch.tensor(cfg.transforms.std)
            filler = filler.view(-1, 1, 1).to(modality_1.device).unsqueeze(0)

            if cfg.transforms.hardcoded_rot != None:
                modality_1_noise = TF.affine(modality_1, harcoded_rot, [0., 0.], 1.0, 0.0, interpolation=TF.InterpolationMode.BILINEAR)
            else:
                all_modality_1_noise = []
                for i in range(len(batch["noise_params"]["angle"])):
                    modality_1_noise = TF.affine(modality_1[i], batch["noise_params"]["angle"][i].item(), 
                                            [batch["noise_params"]["translate"][0][i].item(), batch["noise_params"]["translate"][1][i].item()], 
                                            batch["noise_params"]["scale"][i].item(), batch["noise_params"]["shear"][i].item(), interpolation=TF.InterpolationMode.BILINEAR)
                    all_modality_1_noise.append(modality_1_noise)
                modality_1_noise = torch.stack(all_modality_1_noise)

            modality_1_noise = torch.where(~mask_img.repeat(1, modality_1_noise.shape[1], 1, 1).bool(), filler, modality_1_noise)

            batch["modality_1_noise"] = modality_1_noise
            if cfg.transforms.hardcoded_rot != None:
                batch["noise_params"] = {
                    "angle": torch.tensor([harcoded_rot]* batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device),
                    "translate": [torch.zeros(batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device),
                                            torch.zeros(batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device)],
                    "scale": torch.tensor([1.0]* batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device),
                    "shear": torch.tensor([0.0]* batch["noise_params"]["angle"].shape[0], device=batch["noise_params"]["angle"].device),
                    "image_size": batch["noise_params"]["image_size"],
                    "valid_mask": mask_img,
                }
            embeddings_1_noisy = model.get_embeddings(modality_1_noise)
            masks_noisy = TF.resize(mask_img, size=embeddings_1_noisy.shape[-2:], interpolation=TF.InterpolationMode.NEAREST)

            wrapped_embeddings_1, warp_mask_1 = warp_feature_map_batch(embeddings_1_noisy, batch["noise_params"], image_size=batch["noise_params"]["image_size"], 
                                                 align_corners=False, og_valid_mask=masks_noisy, return_mask=True, return_grid=False)
        
        elif cfg.mode == "post":
            embeddings_1 = model.get_embeddings(modality_1)
            # Scaled noise params
            # Feature -> image scaling
            Sx = embeddings_1.shape[-2] / float(batch["noise_params"]["image_size"][0][0]) 
            Sy = embeddings_1.shape[-1] / float(batch["noise_params"]["image_size"][1][0])
            scaled_noise_params = {
                "angle": batch["noise_params"]["angle"],
                "translate": [batch["noise_params"]["translate"][0] * Sx, batch["noise_params"]["translate"][1] * Sy],
                "scale": batch["noise_params"]["scale"],
                "shear": batch["noise_params"]["shear"],
                "image_size": batch["noise_params"]["image_size"],
            }
            embeddings_1_noisy = []
            masks_noisy = []
            for i in range(len(batch["noise_params"]["angle"])):
                mask_init = TF.resize(batch["transform_params"]["valid_mask"][i], size=embeddings_1[i].shape[-2:], interpolation=TF.InterpolationMode.NEAREST)
                # mask_noise = TF.resize(batch["noise_params"]["valid_mask"][i], size=embeddings_1[i].shape[-2:], interpolation=TF.InterpolationMode.NEAREST)
                mask = mask_init #* mask_noise
                mask = TF.affine(mask, angle=scaled_noise_params["angle"][i].item(), 
                                            translate=[scaled_noise_params["translate"][0][i].item(), scaled_noise_params["translate"][1][i].item()], 
                                            scale=scaled_noise_params["scale"][i].item(), 
                                            shear=scaled_noise_params["shear"][i].item(),
                                            interpolation=TF.InterpolationMode.NEAREST)
                em1_noisy = TF.affine(embeddings_1[i], angle=scaled_noise_params["angle"][i].item(), 
                                            translate=[scaled_noise_params["translate"][0][i].item(), scaled_noise_params["translate"][1][i].item()], 
                                            scale=scaled_noise_params["scale"][i].item(), 
                                            shear=scaled_noise_params["shear"][i].item(),
                                            interpolation=TF.InterpolationMode.BILINEAR)
                mask = torch_erode(mask, kernel_size=3, iterations=1)
                masks_noisy.append(mask)
                embeddings_1_noisy.append(em1_noisy)
            embeddings_1_noisy = torch.stack(embeddings_1_noisy)
            masks_noisy = torch.stack(masks_noisy)
            batch["noise_params"]["angle"] = scaled_noise_params["angle"]
            batch["noise_params"]["translate"] = scaled_noise_params["translate"]
            batch["noise_params"]["scale"] = scaled_noise_params["scale"]
            batch["noise_params"]["shear"] = scaled_noise_params["shear"]

            wrapped_embeddings_1, warp_mask_1 = warp_feature_map_batch(embeddings_1_noisy, batch["noise_params"], image_size=batch["noise_params"]["image_size"], 
                                                 align_corners=False, og_valid_mask=masks_noisy, return_mask=True, return_grid=False)
        else:
            embeddings_1_noisy = model.get_embeddings(modality_1_noise)
            params_noise = batch["noise_params"]
            warp_mask_1 = None
        
            if params_noise["angle"].abs().sum() > 1:
                # We are in medium or hard difficulty
                wrapped_embeddings_1, warp_mask_1 = warp_feature_map_batch(
                            embeddings_1_noisy, params_noise, image_size=params_noise["image_size"], 
                            align_corners=False, og_valid_mask=params_noise["valid_mask"], return_mask=True
                        )
                warp_mask_1 = torch_erode(warp_mask_1, kernel_size=3, iterations=1)
                warp_mask_1 = TF.resize(warp_mask_1, size=wrapped_embeddings_1.shape[-2:], interpolation=TF.InterpolationMode.NEAREST)
            else:
                wrapped_embeddings_1 = embeddings_1_noisy
        
        # Debug visualization if enabled
        if cfg is not None and getattr(cfg, 'viz_debug', False):
            # Create output directory for debug visualizations
            debug_dir = os.path.join(os.path.dirname(to_absolute_path(cfg.logging.output_csv)), "debug_viz")
            os.makedirs(debug_dir, exist_ok=True)
            
            save_path = os.path.join(debug_dir, f"batch_{batch_idx:04d}_debug.png")
            visualize_debug_images(modality_1, modality_1_noise, warp_mask_1, save_path)

        # TODO: Remove this
        # Show the modality 0 and modality 1 with the warp mask using matplotlib plt.imshow
        # The figure will have two rows and two columns
        # The first row will be modality 0 and the second row will be modality 1
        # The first column will have the original images
        # The second column will have the warp_mask_1 * resized_mask_0 and warp_mask_1 respectively
        # for id in range(modality_0.shape[0]):
        #     # Denormalize images for visualization
        #     mean = torch.tensor(cfg.transforms.mean).view(3, 1, 1)
        #     std = torch.tensor(cfg.transforms.std).view(3, 1, 1)
            
        #     modality_0_denorm = modality_0[id].cpu() * std + mean
        #     modality_1_denorm = modality_1_noise[id].cpu() * std + mean
            
        #     # Clamp to valid range [0, 1] and convert to numpy
        #     modality_0_denorm = torch.clamp(modality_0_denorm, 0, 1)
        #     modality_1_denorm = torch.clamp(modality_1_denorm, 0, 1)
            
        #     modality_0_show = modality_0_denorm.permute(1, 2, 0).numpy()
        #     modality_1_show = modality_1_denorm.permute(1, 2, 0).numpy()
            
        #     # Create the 2x2 subplot figure
        #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
        #     # First row - modality 0
        #     axes[0, 0].imshow(modality_0_show)
        #     axes[0, 0].set_title('Modality 0 - Original')
        #     axes[0, 0].axis('off')
            
        #     # Second row - modality 1  
        #     axes[1, 0].imshow(modality_1_show)
        #     axes[1, 0].set_title('Modality 1 - Noise')
        #     axes[1, 0].axis('off')
            
        #     # Compute resized_mask_0 for the visualization
        #     resized_mask_0 = TF.resize(batch["transform_params"]["valid_mask"], size=embeddings_0.shape[-2:], 
        #                     interpolation=TF.InterpolationMode.NEAREST)
            
        #     # First column, second row - modality 0 with combined mask applied
        #     if warp_mask_1 is not None:
        #         warp_mask_1_show = torch_erode(warp_mask_1[id], kernel_size=3, iterations=1)
        #         # Apply combined mask to modality 0
        #         combined_mask = (resized_mask_0[id] * warp_mask_1_show)
        #         combined_mask_img = TF.resize(combined_mask, size=modality_0_denorm.shape[-2:], interpolation=TF.InterpolationMode.BILINEAR)
        #         modality_0_masked = modality_0_denorm * combined_mask_img.cpu()
        #         modality_0_masked_show = modality_0_masked.permute(1, 2, 0).numpy()
        #         axes[0, 1].imshow(modality_0_masked_show)
        #         axes[0, 1].set_title('Modality 0 with (warp_mask_1 * resized_mask_0)')
        #         axes[0, 1].axis('off')
                
        #         # Second column, second row - modality 1 with warp_mask_1 applied
        #         warp_mask_1_expanded = warp_mask_1_show
        #         warp_mask_1_img = TF.resize(warp_mask_1_expanded, size=modality_1_denorm.shape[-2:], interpolation=TF.InterpolationMode.BILINEAR)
        #         modality_1_masked = modality_1_denorm * warp_mask_1_img.cpu()
        #         modality_1_masked_show = modality_1_masked.permute(1, 2, 0).numpy()
        #         axes[1, 1].imshow(modality_1_masked_show)
        #         axes[1, 1].set_title('Modality 1 with warp_mask_1')
        #         axes[1, 1].axis('off')
        #     else:
        #         # If warp_mask_1 is None, apply only resized_mask_0 to modality 0
        #         resized_mask_0_expanded = resized_mask_0[id].unsqueeze(0).expand_as(modality_0_denorm)
        #         modality_0_masked = modality_0_denorm * resized_mask_0_expanded.cpu()
        #         modality_0_masked_show = modality_0_masked.permute(1, 2, 0).numpy()
        #         axes[0, 1].imshow(modality_0_masked_show)
        #         axes[0, 1].set_title('Modality 0 with resized_mask_0 (warp_mask_1 is None)')
        #         axes[0, 1].axis('off')
                
        #         # Show original modality 1 since no warp_mask_1
        #         axes[1, 1].imshow(modality_1_show)
        #         axes[1, 1].set_title('Modality 1 - Original (no warp_mask_1)')
        #         axes[1, 1].axis('off')
            
        #     plt.tight_layout()
            
        #     # Show the figure
        #     plt.show()

        resized_mask_selected = TF.resize(batch["noise_params"]["valid_mask"], size=embeddings_0.shape[-2:], interpolation=TF.InterpolationMode.NEAREST)
        accs = model.compute_retrieval_accuracy2D(embeddings_0, wrapped_embeddings_1, 
                                                    sample_percentage=0.5, topk=3, distance_th=3.0, 
                                                    valid_mask=(warp_mask_1 * resized_mask_selected).bool())
        loss = model.loss_fn(embeddings_0, wrapped_embeddings_1, warp_mask_1)

        metrics = {
            "val_loss": float(loss.item()),
            "val_acc_all": float(accs["acc"].item()),
            "val_acc_self": float(accs["acc_self"].item()),
            "val_acc_others": float(accs["acc_others"].item()),
        }

        if cfg.tta != 1 and cfg.mode in ["gt_tta", "tta"]:
            metrics["tta_accuracy"] = float(accuracy_tta.item())
        else:
            metrics["tta_accuracy"] = 1.0

        print("Metrics: ", metrics)
        
        # Add Euclidean distance and distance threshold metrics if they exist
        if "euclidean_distance" in accs:
            metrics["val_euclidean_distance"] = float(accs["euclidean_distance"].item())
        
        # Add distance threshold metrics if they exist
        for key in accs.keys():
            if key.startswith("distance_below_"):
                metrics[f"val_{key}"] = float(accs[key].item())
        
        # Add top-k metrics if they exist
        for key in accs.keys():
            if key.startswith("acc_top"):
                metrics[f"val_{key}"] = float(accs[key].item())

    return metrics


@hydra.main(config_path="../configs", config_name="val_sensitivity", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_csv_abs = to_absolute_path(cfg.logging.output_csv)
    os.makedirs(os.path.dirname(output_csv_abs), exist_ok=True)

    # Resolve device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(cfg, "runtime") and getattr(cfg.runtime, "device", None) in ("cuda", "cpu"):
        device = cfg.runtime.device

    # Load model
    if not cfg.model.checkpoint:
        raise ValueError("model.checkpoint must be provided (path to .ckpt)")
    # model = load_contrastive_model_from_checkpoint(to_absolute_path(cfg.model.checkpoint))
    model = ContrastiveLearningModule.load_from_checkpoint(checkpoint_path=to_absolute_path(cfg.model.checkpoint), 
                                                            map_location=device, load_dino_weights=False)
    model.to(device)
    model.eval()

    # Dataset & loader
    val_dataset = create_val_dataset(cfg)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.val.batch_size),
        shuffle=False,
        num_workers=int(cfg.val.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    # Iterate and collect metrics per batch
    per_batch: List[Dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        # Move everything in the batch to the device
        batch = move_to_device(batch, device)
        metrics = compute_batch_metrics(model, batch, cfg, batch_idx)
        per_batch.append(metrics)

    # Aggregate metrics (average and std)
    df = pd.DataFrame(per_batch)
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)
    summary_row = {k: float(v) for k, v in means.to_dict().items()}
    summary_row["batches"] = len(per_batch)
    std_row = {f"std_{k}": float(v) for k, v in stds.to_dict().items()}
    # Combined summary dict with both means and stds
    combined_summary = {**summary_row, **std_row}

    # Display and save
    print("Validation metrics per batch:")
    print(df)
    print("\nAverages:")
    print(summary_row)
    print("\nStandard deviations:")
    print(std_row)

    df.to_csv(output_csv_abs, index=False)

    # Also save aggregates (mean and std) as a small CSV next to the main CSV
    agg_csv_path = os.path.splitext(output_csv_abs)[0] + "_agg.csv"
    agg_df = pd.DataFrame({"mean": means, "std": stds})
    agg_df.to_csv(agg_csv_path)

    # Also save a small JSON summary next to CSV
    json_path = os.path.splitext(output_csv_abs)[0] + "_summary.json"
    try:
        import json
        with open(json_path, "w") as f:
            json.dump(combined_summary, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save summary JSON: {e}")


if __name__ == "__main__":
    main()
