import torch
import torch.nn as nn
import torch.nn.functional as F

class SNAPContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # Learnable temperature parameter (initialized as log(tau) often helps stability, 
        # but user requested direct float init. We wrap it as a Parameter.)
        self.temperature = nn.Parameter(torch.tensor(temperature)) 

    def normalize_coords(self, coords: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Normalize pixel coordinates from [0, W-1] / [0, H-1] to [-1, 1] for grid_sample.
        Args:
            coords: (..., 2) where last dim is (x, y)
        """
        # clone to ensure we don't modify inputs in place
        norm_coords = coords.clone()
        norm_coords[..., 0] = 2.0 * norm_coords[..., 0] / (W - 1) - 1.0
        norm_coords[..., 1] = 2.0 * norm_coords[..., 1] / (H - 1) - 1.0
        return norm_coords

    def compute_score(
        self, 
        map_features: torch.Tensor, 
        query_features: torch.Tensor, 
        sampling_coords: torch.Tensor, 
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the alignment score E(T).
        """
        C, H_fp, W_fp = map_features.shape
        
        # --- Ensure coords are Float before normalization ---
        sampling_coords = sampling_coords.float() 
        
        # 1. Normalize coordinates to [-1, 1] for grid_sample
        grid = self.normalize_coords(sampling_coords, H_fp, W_fp)
        
        # 2. Sample Reference Map M^R at projected locations
        # map_features needs shape (1, C, H, W) for grid_sample
        sampled_map_feats = F.grid_sample(
            map_features.unsqueeze(0).expand(query_features.shape[0], -1, -1, -1), 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )

        # 3. L2 Normalize features (Featuremetric scoring)
        # Note: Ensure query_features is also float to avoid type mismatch in dot product
        query_features = F.normalize(query_features.float(), p=2, dim=1)
        sampled_map_feats = F.normalize(sampled_map_feats, p=2, dim=1)

        # ... rest of the function remains the same ...
        
        # 4. Compute Cosine Similarity
        similarity = torch.sum(query_features * sampled_map_feats, dim=1)
        
        # 5. Apply ReLU
        similarity = F.relu(similarity)
        
        # 6. Apply Valid Mask
        mask = valid_mask.squeeze(1).float()
        masked_sim = similarity * mask
        
        # 7. Average over valid pixels
        score_per_sample = masked_sim.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
        
        return score_per_sample

    def forward(
        self, 
        floorplan_features: torch.Tensor, 
        pos_sampling_fustrums: list, 
        neg_sampling_fustrums: list
    ) -> torch.Tensor:
        """
        Args:
            floorplan_features: (B, C, H_fp, W_fp)
            pos_sampling_fustrums: List[FrustumData] of length B. 
                                   Each item has N images.
            neg_sampling_fustrums: List[FrustumData] of length B. 
                                   Each item has N images and K negatives per image.
        Returns:
            Scalar Loss
        """
        batch_size = floorplan_features.shape[0]
        total_loss = 0.0
        
        # Iterate over batch elements because FrustumData is a list of objects
        # and grid sizes / masks might arguably differ (though usually fixed).
        for b in range(batch_size):
            # --- Prepare Data for Batch 'b' ---
            map_feat = floorplan_features[b] # (C, H_fp, W_fp)
            
            # Positive Data
            pos_item = pos_sampling_fustrums[b]
            pos_feats = pos_item.features          # (N, C, H_g, W_g)
            pos_coords = pos_item.coords_proj_xy   # (N, H_g, W_g, 2)
            pos_mask = pos_item.valid_mask_xy      # (N, 1, H_g, W_g)
            
            # Negative Data
            neg_item = neg_sampling_fustrums[b]
            # neg_item.features is (N, C, H_g, W_g) - query feats don't change
            neg_coords = neg_item.coords_proj_xy   # (N, K, H_g, W_g, 2)
            neg_mask = neg_item.valid_mask_xy      # (N, K, 1, H_g, W_g)
            
            N_imgs, K_negs, _, _, _ = neg_coords.shape
            
            # --- 1. Compute Positive Scores ---
            # Result: (N_imgs,)
            pos_scores = self.compute_score(map_feat, pos_feats, pos_coords, pos_mask)
            
            # --- 2. Compute Negative Scores ---
            # We need to flatten N and K to process efficiently
            # Expand features to match negative samples: (N, 1, C...) -> (N, K, C...)
            neg_feats_expanded = neg_item.features.unsqueeze(1).expand(-1, K_negs, -1, -1, -1)
            
            # Flatten to (N*K, ...)
            flat_neg_feats = neg_feats_expanded.reshape(-1, *pos_feats.shape[1:])
            flat_neg_coords = neg_coords.reshape(-1, *pos_coords.shape[1:])
            flat_neg_mask = neg_mask.reshape(-1, *pos_mask.shape[1:])
            
            # Result: (N*K,)
            flat_neg_scores = self.compute_score(map_feat, flat_neg_feats, flat_neg_coords, flat_neg_mask)
            
            # Reshape back to (N, K)
            neg_scores = flat_neg_scores.view(N_imgs, K_negs)
            
            # --- 3. Compute InfoNCE Loss (Eq. 3) ---
            # Logits: Concatenate Positive (N, 1) and Negative (N, K) -> (N, 1+K)
            # Positive sample is at index 0
            logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            
            # Apply temperature
            logits = logits / self.temperature
            
            # Target is always index 0 (the positive sample)
            labels = torch.zeros(N_imgs, dtype=torch.long, device=logits.device)
            
            # Loss for this batch item (averaged over N images)
            loss_b = F.cross_entropy(logits, labels)
            total_loss += loss_b

        # Return average loss over the batch size
        return total_loss / batch_size


class SimplifiedSNAPLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, num_negatives: int = 4096):
        """
        Args:
            temperature: Softmax temperature scaling.
            num_negatives: Number of negative samples to draw from the floorplan 
                           for the contrastive loss (subset for efficiency).
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.num_negatives = num_negatives

    def normalize_coords(self, coords: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Normalize pixel coordinates from [0, W-1] / [0, H-1] to [-1, 1] for grid_sample.
        """
        norm_coords = coords.clone()
        norm_coords[..., 0] = 2.0 * norm_coords[..., 0] / (W - 1) - 1.0
        norm_coords[..., 1] = 2.0 * norm_coords[..., 1] / (H - 1) - 1.0
        return norm_coords

    def forward(
        self, 
        floorplan_features: torch.Tensor, 
        pos_sampling_fustrums: list
    ) -> torch.Tensor:
        """
        Args:
            floorplan_features: (B, C, H_fp, W_fp) Reference map features.
            pos_sampling_fustrums: List of SceneFrustums (length B).
                                   Each item has N images with:
                                   - features: (N, C, H_g, W_g)
                                   - coords_proj_xy: (N, H_g, W_g, 2)
                                   - valid_mask_xy: (N, 1, H_g, W_g)
        """
        batch_size = floorplan_features.shape[0]
        total_loss = 0.0
        device = floorplan_features.device

        for b in range(batch_size):
            # --- 1. Unpack Data for Batch 'b' ---
            # Map: (C, H_fp, W_fp)
            map_feat = floorplan_features[b]
            C, H_fp, W_fp = map_feat.shape
            
            # Frustum: Query Features and GT Coordinates
            # We assume these are already computed/transformed for the GT pose
            frustum = pos_sampling_fustrums[b]
            query_feat = frustum.features          # (N, C, H_g, W_g)
            query_coords = frustum.coords_proj_xy  # (N, H_g, W_g, 2)
            valid_mask = frustum.valid_mask_xy     # (N, 1, H_g, W_g)

            # --- 2. L2 Normalize Features ---
            # Normalize map features once (C, H_fp, W_fp)
            map_feat_norm = F.normalize(map_feat.float(), p=2, dim=0)
            
            # Normalize query features (N, C, H_g, W_g)
            query_feat_norm = F.normalize(query_feat.float(), p=2, dim=1)

            # --- 3. Sample Positives (GT Alignment) ---
            # We need to sample the map at the projected query locations.
            # Grid sample expects (N, C, H_in, W_in) input and (N, H_out, W_out, 2) grid
            
            # Prepare Grid: Normalize coords to [-1, 1]
            # (N, H_g, W_g, 2)
            grid = self.normalize_coords(query_coords, H_fp, W_fp).to(torch.float32)
            
            # Prepare Map: Expand to match N images (N, C, H_fp, W_fp)
            map_expanded = map_feat_norm.unsqueeze(0).expand(query_feat.shape[0], -1, -1, -1)
            
            # Sample: (N, C, H_g, W_g)
            positive_keys = F.grid_sample(
                map_expanded, 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            )

            # --- 4. Prepare Negatives (Random Subset from Floorplan) ---
            # We treat the floorplan as a bag of features. We sample a subset of "random columns"
            # to serve as negative keys for ALL queries in this batch item.
            
            # Flatten map spatial dims: (C, H_fp * W_fp) -> (C, TotalPixels)
            map_flat = map_feat_norm.view(C, -1)
            total_pixels = map_flat.shape[1]
            
            # Select random indices
            num_negs = min(self.num_negatives, total_pixels)
            perm = torch.randperm(total_pixels, device=device)[:num_negs]
            
            # Gather negatives: (C, NumNegs)
            negative_keys = map_flat[:, perm]

            # --- 5. Compute Contrastive Loss (Pixel-wise) ---
            # We only want to compute loss for VALID query pixels (those that fall inside the map/fov).
            
            # Flatten Query and Positives to lists of pixels
            # Permute to (N, H_g, W_g, C) -> Flatten to (TotalQueryPixels, C)
            q_flat = query_feat_norm.permute(0, 2, 3, 1).reshape(-1, C)
            p_flat = positive_keys.permute(0, 2, 3, 1).reshape(-1, C)
            m_flat = valid_mask.permute(0, 2, 3, 1).reshape(-1) # (TotalQueryPixels,)

            # Filter valid pixels
            valid_idx = torch.nonzero(m_flat > 0.5).squeeze()
            
            # Optimization: If no valid pixels, skip
            if valid_idx.numel() == 0:
                continue

            q_selected = q_flat[valid_idx] # (NumValid, C)
            p_selected = p_flat[valid_idx] # (NumValid, C)

            # A. Positive Scores: Dot Product (Cosine Sim)
            # (NumValid, C) * (NumValid, C) -> (NumValid,)
            logits_pos = torch.sum(q_selected * p_selected, dim=1, keepdim=True) # (NumValid, 1)

            # B. Negative Scores: Matrix Multiply
            # (NumValid, C) @ (C, NumNegs) -> (NumValid, NumNegs)
            logits_neg = torch.matmul(q_selected, negative_keys)

            # --- 6. InfoNCE Calculation ---
            # Concatenate: [Pos, Negs] -> (NumValid, 1 + NumNegs)
            logits = torch.cat([logits_pos, logits_neg], dim=1)
            
            # Apply Temperature
            logits = logits / self.temperature

            # Target is always index 0 (the positive key)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            
            loss_b = F.cross_entropy(logits, labels)
            total_loss += loss_b

        return total_loss / batch_size

class FrustumRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def normalize_coords(self, coords: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Normalize pixel coordinates from [0, W-1] / [0, H-1] to [-1, 1] for grid_sample.
        Args:
            coords: (..., 2) Pixel coordinates (x, y)
        """
        norm_coords = coords.clone().float()
        norm_coords[..., 0] = 2.0 * norm_coords[..., 0] / (W - 1) - 1.0
        norm_coords[..., 1] = 2.0 * norm_coords[..., 1] / (H - 1) - 1.0
        return norm_coords

    def forward(
        self, 
        floorplan_features: torch.Tensor, 
        pos_sampling_fustrums: list
    ) -> torch.Tensor:
        """
        Computes regression loss (MSE) between the predicted frustum features and 
        the features on the floorplan at the ground-truth location.

        Args:
            floorplan_features: (B, C, H_fp, W_fp) Reference map features.
            pos_sampling_fustrums: List of SceneFrustums (length B).
                                   Each item has N images with:
                                   - features: (N, C, H_g, W_g) -> The predicted features
                                   - coords_proj_xy: (N, H_g, W_g, 2) -> Locations on floorplan
                                   - valid_mask_xy: (N, 1, H_g, W_g) -> Valid in camera view
        """
        batch_size = floorplan_features.shape[0]
        total_loss = 0.0
        device = floorplan_features.device

        for b in range(batch_size):
            # --- 1. Unpack Data for Batch 'b' ---
            map_feat = floorplan_features[b] # (C, H_fp, W_fp)
            C, H_fp, W_fp = map_feat.shape
            
            frustum = pos_sampling_fustrums[b]
            pred_features = frustum.features       # (N, C, H_g, W_g)
            target_coords = frustum.coords_proj_xy # (N, H_g, W_g, 2) (Pixel units)
            camera_mask = frustum.valid_mask_xy    # (N, 1, H_g, W_g)

            # --- 2. Sample Target Features from Floorplan ---
            
            # Normalize coordinates to [-1, 1]
            # (N, H_g, W_g, 2)
            grid = self.normalize_coords(target_coords, H_fp, W_fp).to(torch.float32)
            
            # Prepare Map: Expand to match N images (1, C, H_fp, W_fp) -> (N, C, H_fp, W_fp)
            # grid_sample needs batch dimension to match grid
            map_expanded = map_feat.unsqueeze(0).expand(pred_features.shape[0], -1, -1, -1)
            
            # Sample features at the GT locations
            # Result: (N, C, H_g, W_g)
            target_features = F.grid_sample(
                map_expanded, 
                grid, 
                mode='bilinear', 
                padding_mode='border', 
                align_corners=True
            )

            # --- 3. Detach Target ---
            # As requested: floorplan features are fixed anchors. 
            # We don't want to pull the floorplan towards the bad frustum prediction.
            target_features = target_features.detach()

            # --- 4. Compute Validity Mask ---
            # A pixel is valid if:
            # A) It is visible in the camera (camera_mask)
            # B) It projects INSIDE the floorplan boundaries (grid values in [-1, 1])
            
            # Check B: Floorplan bounds
            in_bounds_x = (grid[..., 0] >= -1) & (grid[..., 0] <= 1)
            in_bounds_y = (grid[..., 1] >= -1) & (grid[..., 1] <= 1)
            floorplan_mask = (in_bounds_x & in_bounds_y).unsqueeze(1) # (N, 1, H_g, W_g)
            
            # Mask zero frustum features
            zero_mask = (pred_features.abs().sum(dim=1, keepdim=True) != 0)

            # Combined mask (N, 1, H_g, W_g)
            final_mask = camera_mask * floorplan_mask.float() * zero_mask.float()

            # Optimization: If no valid pixels in this scene, skip
            if final_mask.sum() == 0:
                continue

            # --- 5. Compute Cosine Similarity Loss ---
            # We compute cosine similarity per pixel/channel, then mean over valid pixels
            pred = F.normalize(pred_features, dim=1)
            tgt  = F.normalize(target_features, dim=1)
            # tgt = target_features # Remove normalization to encode uncertainty 
            cos_sim = (pred * tgt).sum(dim=1, keepdim=True)  # (N,1,H,W)
            loss_map = 1.0 - cos_sim
            loss_b = (loss_map * final_mask).sum() / (final_mask.sum() + 1e-6)
            
            total_loss += loss_b

        return total_loss / batch_size