import torch
import torch.nn as nn
import torch.nn.functional as F

class RoMaV2Loss(nn.Module):
    def __init__(self, regression_weight=1.0, confidence_weight=1.0):
        super().__init__()
        self.regression_weight = regression_weight
        self.confidence_weight = confidence_weight

    def robust_l1_loss(self, pred, target, mask=None, epsilon=1e-6):
        """
        Computes L1 loss only on valid pixels.
        """
        loss = torch.abs(pred - target)
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + epsilon)
        return loss.mean()

    def compute_quadrant_indices(self, pred_warp, gt_warp, gt_mask, eps=1e-6):
        """
        pred_warp: [B,H,W,2]
        gt_warp:   [B,H,W,2]
        gt_mask:   [B,H,W,1] or [B,H,W]

        Returns:
            quadrant_indices: [B,H,W] with values ∈ {0,1,2,3} or -1 for ignore
        """
        # Ensure mask shape is [B,H,W]
        if gt_mask.ndim == 4:
            gt_mask = gt_mask[..., 0]

        dx = gt_warp[..., 0] - pred_warp[..., 0]
        dy = gt_warp[..., 1] - pred_warp[..., 1]

        # If offset is too small → ignore (model can't infer quadrant)
        small_disp = (dx.abs() < eps) & (dy.abs() < eps)

        # Quadrant assignments
        q0 = (dx < 0) & (dy < 0)                 # top-left
        q1 = (dx > 0) & (dy < 0)                 # top-right
        q2 = (dx < 0) & (dy > 0)                 # bottom-left
        q3 = (dx > 0) & (dy > 0)                 # bottom-right

        quadrant = torch.full_like(dx, -1, dtype=torch.long)  # start with ignore_index

        quadrant[q0] = 0
        quadrant[q1] = 1
        quadrant[q2] = 2
        quadrant[q3] = 3

        # Mask out invalid areas
        quadrant[gt_mask == 0] = -1
        quadrant[small_disp] = -1

        return quadrant

    def forward(self, out, gt_warp, gt_mask):
        """
        Args:
            out: Dictionary output from RoMa model.
            gt_warp: Tensor [B, H, W, 2] in range [-1, 1].
            gt_mask: Tensor [B, H, W, 1], 1 if pixel has valid correspondence.
        """
        total_loss = 0
        metrics = {}

        # Keys in RoMa output that contain warp predictions
        # Usually: 'matcher' (coarse), 'refiner_4', 'refiner_2', 'refiner_1', 'warp' (final)
        # We iterate over them to apply deep supervision.
        
        # 1. Main Loss (Final Warp)
        # -------------------------
        # out["warp"] is a Tensor [B, H, W, 2]
        pred_warp_final = out["warp_AB"]
        
        # Ensure GT matches prediction resolution (should be same for final)
        if pred_warp_final.shape[1:3] != gt_warp.shape[1:3]:
            curr_gt = F.interpolate(gt_warp.permute(0,3,1,2), 
                                    size=pred_warp_final.shape[1:3], 
                                    mode='bilinear').permute(0,2,3,1)
            curr_mask = F.interpolate(gt_mask.permute(0,3,1,2), 
                                      size=pred_warp_final.shape[1:3], 
                                      mode='nearest').permute(0,2,3,1)
        else:
            curr_gt = gt_warp
            curr_mask = gt_mask

        l_reg_final = self.robust_l1_loss(pred_warp_final, curr_gt, curr_mask)
        total_loss += self.regression_weight * l_reg_final
        metrics['loss_warp_final'] = l_reg_final.item()

        # 2. Auxiliary Losses (Matcher & Refiners)
        # ----------------------------------------
        aux_keys = ['matcher', 'refiner_4', 'refiner_2', 'refiner_1']
        
        for key in aux_keys:
            if key not in out:
                continue
                
            pred_dict = out[key]
            if 'warp_AB' not in pred_dict:
                continue
                
            pred_warp = pred_dict['warp_AB']
            
            # Interpolate GT to current scale
            # Note: RoMa 'matcher' output is usually very coarse (e.g. 14x14 patches)
            # We align GT to prediction size
            H_curr, W_curr = pred_warp.shape[1], pred_warp.shape[2]
            
            curr_gt = F.interpolate(gt_warp.permute(0,3,1,2), 
                                    size=(H_curr, W_curr), 
                                    mode='bilinear', align_corners=False).permute(0,2,3,1)
            
            curr_mask = F.interpolate(gt_mask.permute(0,3,1,2), 
                                      size=(H_curr, W_curr), 
                                      mode='nearest').permute(0,2,3,1)
            
            # Regression Loss
            l_reg = self.robust_l1_loss(pred_warp, curr_gt, curr_mask)
            total_loss += (self.regression_weight * 0.5) * l_reg # Lower weight for aux
            metrics[f'loss_reg_{key}'] = l_reg.item()
            
            # Confidence Loss (Optional but recommended)
            # If the model predicts confidence, we supervise it with the valid mask
            if 'confidence_AB' in pred_dict and pred_dict['confidence_AB'] is not None:
                pred_conf = pred_dict['confidence_AB']  # [B,H,W,4] logits

                # Compute GT quadrant labels
                gt_quadrants = self.compute_quadrant_indices(pred_warp, curr_gt, curr_mask)

                # Convert to shape B×C×H×W for cross-entropy
                pred_conf_ce = pred_conf.permute(0, 3, 1, 2)   # BHWC → B C H W

                # CrossEntropyLoss expects ignore_index for invalid pixels
                ce_loss = F.cross_entropy(
                    pred_conf_ce,
                    gt_quadrants,
                    ignore_index=-1
                )

                total_loss += self.confidence_weight * ce_loss
                metrics[f'loss_conf_{key}'] = ce_loss.item()

        return total_loss, metrics


class RoMaV1Loss(nn.Module):
    def __init__(self, weights=None, confidence_weight=0.01, alpha=1.0, c=1e-3):
        """
        Args:
            weights (dict): Weights for each scale {16: w1, 8: w2, ...}.
            confidence_weight (float): Weight for the certainty (BCE) loss. 
                                       Official RoMa V1 uses 0.01.
            alpha (float): Power parameter for the robust regression loss.
            c (float): Scale parameter for the robust regression loss.
        """
        super().__init__()
        # Default weights if not provided
        self.weights = weights if weights is not None else {16: 1.0, 8: 1.0, 4: 1.0, 2: 1.0, 1: 1.0}
        self.confidence_weight = confidence_weight
        self.alpha = alpha
        self.c = c

    def robust_regression_loss(self, pred_flow, target_warp, mask, scale):
        """
        Robust regression: loss = (c*s)^a * ((epe / (c*s))^2 + 1)^(a/2)
        """
        # 1. Compute EPE (End-Point Error)
        # pred_flow: [B, H, W, 2], target_warp: [B, H, W, 2]
        diff = pred_flow - target_warp
        epe = torch.norm(diff, dim=-1)

        # 2. Robust Loss Formula
        # 'cs' scales the robustness threshold based on resolution scale
        cs = self.c * scale
        loss_map = (cs ** self.alpha) * ((epe / cs)**2 + 1) ** (self.alpha / 2)

        # 3. Apply Mask (Only compute regression on valid ground truth pixels)
        if mask is not None:
            if mask.ndim == 4:
                mask = mask.squeeze(-1)  # [B, H, W]
            
            loss_map = loss_map * mask
            # Add epsilon to denominator to prevent division by zero
            return loss_map.sum() / (mask.sum() + 1e-6)
        
        return loss_map.mean()

    def forward(self, out, gt_warp, gt_mask):
        """
        Args:
            out: Dict of dicts {scale: {'flow': [B,2,H,W], 'certainty': [B,1,H,W]}}
            gt_warp: Tensor [B, H, W, 2] (normalized [-1, 1])
            gt_mask: Tensor [B, H, W, 1] (1 for valid, 0 for invalid)
        """
        total_loss = 0
        metrics = {}
        
        # Iterate over scales (16 -> 1)
        scales = [16, 8, 4, 2, 1]
        
        for scale in scales:
            if scale not in out:
                continue
            
            # --- 1. Unpack Prediction ---
            pred_dict = out[scale]
            pred_flow = pred_dict['flow']           # [B, 2, H, W]
            pred_certainty = pred_dict['certainty'] # [B, 1, H, W] logits
            
            # --- 2. Align GT to Current Scale ---
            # Current spatial resolution
            H_curr, W_curr = pred_flow.shape[2], pred_flow.shape[3]
            
            # Interpolate GT Warp (Bilinear)
            # Input to interpolate must be [B, C, H, W], so we verify GT shape
            # gt_warp is [B, H, W, 2] -> permute to [B, 2, H, W] for interp
            curr_gt_warp = F.interpolate(
                gt_warp.permute(0, 3, 1, 2), 
                size=(H_curr, W_curr), 
                mode='bilinear', 
                align_corners=True
            ).permute(0, 2, 3, 1) # Back to [B, H, W, 2] for loss calc

            # Interpolate GT Mask (Nearest)
            curr_gt_mask = F.interpolate(
                gt_mask.permute(0, 3, 1, 2).float(), 
                size=(H_curr, W_curr), 
                mode='nearest'
            ).permute(0, 2, 3, 1) # Back to [B, H, W, 1]

            # --- 3. Regression Loss (Flow) ---
            # Permute pred_flow from [B, 2, H, W] -> [B, H, W, 2]
            pred_flow_permuted = pred_flow.permute(0, 2, 3, 1)
            
            reg_loss = self.robust_regression_loss(
                pred_flow_permuted, 
                curr_gt_warp, 
                curr_gt_mask, 
                scale
            )
            
            # --- 4. Certainty Loss (BCE) ---
            # pred_certainty: [B, 1, H, W] (logits)
            # curr_gt_mask:   [B, H, W, 1] -> permute to [B, 1, H, W] for BCE
            target_certainty = curr_gt_mask.permute(0, 3, 1, 2)
            
            # Binary Cross Entropy with Logits
            # We supervise certainty everywhere (model should predict 0 for invalid areas)
            cert_loss = F.binary_cross_entropy_with_logits(
                pred_certainty, 
                target_certainty
            )

            # --- 5. Combine Losses ---
            scale_weight = self.weights.get(scale, 1.0)
            
            # Weighted sum for this scale
            # Note: Official code weights certainty by `ce_weight` (usually 0.01)
            loss_scale = reg_loss + (self.confidence_weight * cert_loss)
            
            total_loss += scale_weight * loss_scale
            
            # --- 6. Metrics ---
            metrics[f'loss_reg_{scale}'] = reg_loss.item()
            metrics[f'loss_cert_{scale}'] = cert_loss.item()

        return total_loss, metrics