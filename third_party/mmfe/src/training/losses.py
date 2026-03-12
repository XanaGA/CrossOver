"""
Contrastive learning loss functions.
Includes InfoNCE and other contrastive losses.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import torch.utils.checkpoint as checkpoint


class InfoNCELoss1D(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        reduction="mean",
        epsilon=None,
        max_negatives=None,
        min_negatives=1,   
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.epsilon = epsilon
        self.max_negatives = max_negatives
        self.min_negatives = min_negatives

    def forward(self, embeddings_0, embeddings_1):
        B = embeddings_0.size(0)

        z0 = F.normalize(embeddings_0, dim=1)
        z1 = F.normalize(embeddings_1, dim=1)
        z = torch.cat([z0, z1], dim=0)  # (2B, D)

        sim = torch.matmul(z, z.T) / self.temperature

        diag_mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(diag_mask, float("-inf"))

        idx = torch.arange(2 * B, device=z.device)
        pos_idx = (idx + B) % (2 * B)
        pos_sim = sim[idx, pos_idx]

        # --- Negative mask ---
        neg_mask = ~diag_mask
        neg_mask.scatter_(1, pos_idx.unsqueeze(1), False)

        if self.epsilon is not None:
            hard_mask = sim > (pos_sim.unsqueeze(1) - self.epsilon)
            neg_mask &= hard_mask

        # ---- SAFETY CHECK ----
        num_negs = neg_mask.sum(dim=1)

        # fallback: use all negatives if too few
        fallback = num_negs < self.min_negatives
        if fallback.any():
            neg_mask[fallback] = (~diag_mask & ~F.one_hot(pos_idx, 2 * B).bool())[fallback]

        masked_sim = sim.masked_fill(~neg_mask, float("-inf"))

        if self.max_negatives is not None:
            neg_vals, _ = torch.topk(
                masked_sim,
                k=min(self.max_negatives, masked_sim.size(1)),
                dim=1
            )
        else:
            neg_vals = masked_sim

        # include the positive in the denominator
        logits = torch.cat(
            [pos_sim.unsqueeze(1), neg_vals],
            dim=1
        )

        logsumexp = torch.logsumexp(logits, dim=1)

        losses = -pos_sim + logsumexp

        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses



class InfoNCELoss(nn.Module):
    """
    InfoNCE loss (SimCLR style).

    Args:
        temperature: temperature scalar (float)
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean', is_2d: bool = False, block_size: int = 1, neg_sampling: str = "random", neg_k: int = 0, random_per_row: bool = True):
        super().__init__()
        self.temperature = temperature
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.reduction = reduction
        self.is_2d = is_2d
        self.block_size = block_size
        self._get_infonce = self._get_infonce_block if (block_size > 1 or is_2d) else self._get_infonce_single
        # Base spatial resolution used to define sampling behaviour for 2D embeddings.
        # At 32x32 we use all tokens; at 64x64 we sample ~every 2, at 128x128 every 4, etc.
        self._base_spatial_side = 32
        self.neg_sampling = neg_sampling
        self.neg_k = neg_k
        self.random_per_row = random_per_row

    def forward(self, embeddings_0: torch.Tensor, embeddings_1: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        embeddings_0: (B, D)
        embeddings_1: (B, D)

        Returns:
            loss: scalar (or vector if reduction='none')
        """
        if self.is_2d:
            assert embeddings_0.dim() == 4 and embeddings_1.dim() == 4
        else:
            assert masks is None, "masks are not supported for 1D embeddings"
            assert embeddings_0.dim() == 2 and embeddings_1.dim() == 2
        assert embeddings_0.shape == embeddings_1.shape, "embeddings must have same shape"

        if self.is_2d:
            batch_size, channels, height, width = embeddings_0.shape
            # Flatten spatial dimensions
            embeddings_0 = embeddings_0.permute(0, 2, 3, 1)
            embeddings_0 = embeddings_0.reshape(-1, channels)
            embeddings_1 = embeddings_1.permute(0, 2, 3, 1)
            embeddings_1 = embeddings_1.reshape(-1, channels)
            if masks is not None:
                masks = masks.permute(0, 2, 3, 1)
                masks = masks.reshape(-1, 1)
                masks = torch.cat([masks, masks], dim=0)

        else:
            batch_size, channels = embeddings_0.shape

        # concatenate to get 2B x D or 2BHW x D
        z = torch.cat([embeddings_0, embeddings_1], dim=0)  # (2B, D) or (2BHW, D)

        return self._get_infonce(z, masks)

    def _get_infonce_single(self, z: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        # normalize to unit vectors (defensive: in case inputs are not normalized)
        z = F.normalize(z, p=2, dim=1)

        examples_considered = z.shape[0]
        examples_in_the_batch = examples_considered

        # similarity matrix (2B x 2B)
        sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B) or (2BHW, 2BHW)

        # mask out self-similarities on the diagonal
        diag_mask = torch.eye(examples_considered, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(diag_mask, float('-inf'))  # or a very large negative number

        # positive mask: identity rolled by batch_size (pos_i is at i+batch_size and vice versa)
        # In square doesn't matter which dimension we roll, but in non-square dims it does
        pos_mask = diag_mask.roll(shifts=examples_in_the_batch//2, dims=1)  # (2B, 2B) boolean

        # extract positive similarities (length 2B)
        pos_sim = sim[pos_mask]  # shape (2B,)

        # log-sum-exp denominator for each row
        logsumexp = torch.logsumexp(sim, dim=1)  # (2B,)

        # negative log-likelihood for each sample: -sim_pos + logsumexp
        losses = -pos_sim + logsumexp  # (2B,)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses  # (2B,) vecto

    def _get_infonce_single_block(
            self,
            z: torch.Tensor,
            b: torch.Tensor,
            indices: List[int],
            masks: torch.Tensor = None,
        ) -> torch.Tensor:
        # normalize to unit vectors (defensive: in case inputs are not normalized)
        z = F.normalize(z, p=2, dim=1)

        examples_considered = z.shape[0]
        examples_in_the_batch = b.shape[0]

        # similarity matrix (2B x 2B)
        sim = torch.matmul(z, b.T) / self.temperature  # (2B, 2B) or (2BHW, 2BHW)

        # mask out self-similarities on the diagonal
        diag_mask = torch.zeros_like(sim, dtype=torch.bool, device=sim.device)
        diag_mask[list(range(examples_considered)), indices] = True
        sim = sim.masked_fill(diag_mask, float('-inf'))  # or a very large negative number
        if masks is not None:
            sim = sim.masked_fill(~masks.squeeze()[None,...].bool().repeat(examples_considered,1), float('-inf'))

        # positive mask: identity rolled by batch_size (pos_i is at i+batch_size and vice versa)
        # In square doesn't matter which dimension we roll, but in non-square dims it does
        pos_mask = diag_mask.roll(shifts=examples_in_the_batch//2, dims=1)  # (2B, 2B) boolean

        # extract positive similarities (length 2B)
        pos_sim = sim[pos_mask]  # shape (2B,)

        del z, b, diag_mask
        torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # NEGATIVE SAMPLING
        # ----------------------------------------------------------------------

        if self.neg_sampling.lower() == "hard":
            if self.neg_k == 0:
                self.neg_k = examples_in_the_batch // 2
            
            # Ensure we don't exceed available negatives (Total - Self - Pos)
            K = min(self.neg_k, examples_in_the_batch - 2)

            # 1. HIDE POSITIVES IN-PLACE
            # We already cached the values in 'pos_sim', so we can overwrite 
            # them in 'sim' with -inf to hide them from topk.
            sim.masked_fill_(pos_mask, float('-inf'))

            # 2. FIND HARD NEGATIVES
            # Now 'sim' only contains valid negatives (and -inf elsewhere)
            _, neg_idx = torch.topk(sim, k=K, dim=1)

            # 3. CREATE MASK (bool is 1 byte, much smaller than float clone)
            sampled_mask = torch.zeros_like(sim, dtype=torch.bool)
            sampled_mask.scatter_(1, neg_idx, True)

            # 4. APPLY NEGATIVE MASK IN-PLACE
            # Set everything that isn't a top-k negative to -inf
            sim.masked_fill_(~sampled_mask, float('-inf'))

            # 5. RESTORE POSITIVES IN-PLACE
            # Write the original positive scores back into the matrix
            sim[pos_mask] = pos_sim

            # 6. ASSIGN
            sim_sampled = sim

        elif self.neg_sampling.lower() == "random":
            
            if self.neg_k == 0:
                self.neg_k = examples_in_the_batch // 2
            
            # Safety: Ensure we don't try to sample more negatives than exist.
            # Total - 1 (Self) - 1 (Positive) = Total - 2
            K = min(self.neg_k, examples_in_the_batch - 2)

            # 1. HIDE POSITIVES IN-PLACE
            # Mask out the positive pair temporarily so we don't accidentally pick it 
            # as a random negative.
            sim.masked_fill_(pos_mask, float('-inf'))

            if self.random_per_row:
                # 2A. VECTORIZED RANDOM SAMPLING (No Loop)
                # We generate a random noise matrix of the same size as sim.
                # Conceptually: "Sort by Randomness" instead of "Sort by Similarity".
                noise = torch.rand_like(sim)

                # Ensure we don't select indices that are already -inf (Diagonal or Positive)
                # We set their noise score to -1.0 so they drop to the bottom of topk
                noise.masked_fill_(sim == float('-inf'), -1.0)

                # Select top-K random indices
                _, neg_idx = torch.topk(noise, k=K, dim=1)

                # Create the boolean mask
                sampled_mask = torch.zeros_like(sim, dtype=torch.bool)
                sampled_mask.scatter_(1, neg_idx, True)

            else:
                # 2B. SHARED RANDOM NEGATIVES
                candidates = torch.randperm(sim.size(1), device=sim.device)
                chosen_indices = candidates[:K]

                sampled_mask = torch.zeros_like(sim, dtype=torch.bool)
                sampled_mask[:, chosen_indices] = True

            # 3. APPLY MASK IN-PLACE
            # Keep only the sampled random negatives (and existing -inf stay -inf)
            sim.masked_fill_(~sampled_mask, float('-inf'))

            # 4. RESTORE POSITIVES IN-PLACE
            # Bring back the positive values we cached earlier
            sim[pos_mask] = pos_sim
            
            # 5. ASSIGN
            sim_sampled = sim

        else:
            # No sampling — full InfoNCE
            sim_sampled = sim

        # ----------------------------------------------------------------------
        # Compute the loss
        # ----------------------------------------------------------------------
        logsumexp = torch.logsumexp(sim_sampled, dim=1)   # (N,)
        losses = -pos_sim + logsumexp                     # (N,)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses   # (N,)

    def _get_infonce_block(self, z: torch.Tensor, masks: torch.Tensor = None, sample_percentage: float = 0.5) -> torch.Tensor:
        """
        Blocked InfoNCE computation.

        z: (2N, D)

        Returns: scalar loss (or vector if reduction == 'none')
        """
        if masks is None:
            masks = torch.ones_like(z[:, 0], dtype=torch.bool, device=z.device)
        latents_per_batch = z.shape[0]
        # Sample the percentage of the batch
        if sample_percentage > 0.0:
            num_selected = int(latents_per_batch * sample_percentage)
            indices = np.array(list(range(latents_per_batch)))
            permutation = torch.randperm(len(indices))
            perm_mask = masks.bool().cpu()[permutation].squeeze()
            indices = indices[permutation][perm_mask]
            selected_indices = indices[:num_selected]
        else:
            num_selected = latents_per_batch
            selected_indices = list(range(num_selected))[masks.bool()]

        losses_sum = 0.0
        for block_idx, i in enumerate(range(0, num_selected, self.block_size)):
            # print(f"Processing block {block_idx} of {num_selected // self.block_size}")
            block_indices = selected_indices[i:i+self.block_size]
            z_block = z[block_indices]
            loss = checkpoint.checkpoint(self._get_infonce_single_block, z_block, z, block_indices, masks, use_reentrant=True)
            losses_sum += loss

        if self.reduction == 'mean':
            total_losses = losses_sum / num_selected
        elif self.reduction == 'sum':
            total_losses = losses_sum
        return total_losses


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Alternative implementation of InfoNCE.
    
    Args:
        temperature: Temperature parameter for softmax (default: 0.07)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, embeddings_0: torch.Tensor, embeddings_1: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            embeddings_0: First modality embeddings (B, D)
            embeddings_1: Second modality embeddings (B, D)
            
        Returns:
            Loss value
        """
        batch_size = embeddings_0.size(0)
        
        # Normalize embeddings (should already be normalized, but just in case)
        embeddings_0 = F.normalize(embeddings_0, dim=1)
        embeddings_1 = F.normalize(embeddings_1, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings_0, embeddings_1.T) / self.temperature
        
        # Create labels for positive pairs (diagonal)
        labels = torch.arange(batch_size, device=embeddings_0.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for contrastive learning.
    
    Args:
        margin: Margin for triplet loss (default: 1.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, embeddings_0: torch.Tensor, embeddings_1: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            embeddings_0: First modality embeddings (B, D)
            embeddings_1: Second modality embeddings (B, D)
            
        Returns:
            Loss value
        """
        batch_size = embeddings_0.size(0)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings_0, embeddings_1, p=2)
        
        # Positive pairs are on the diagonal
        positive_dist = torch.diag(dist_matrix)
        
        # For each sample, find hardest negative (closest negative)
        # We need to mask out the positive pairs
        mask = torch.eye(batch_size, device=embeddings_0.device, dtype=torch.bool)
        negative_dist = dist_matrix[~mask].view(batch_size, batch_size - 1)
        hardest_negative = negative_dist.min(dim=1)[0]
        
        # Compute triplet loss
        triplet_loss = F.relu(positive_dist - hardest_negative + self.margin)
        
        if self.reduction == 'mean':
            return triplet_loss.mean()
        elif self.reduction == 'sum':
            return triplet_loss.sum()
        else:
            return triplet_loss


class ContrastiveLoss(nn.Module):
    """
    Wrapper class for different contrastive loss functions.
    
    Args:
        loss_type: Type of loss ('infonce', 'infonce2d', 'ntxent', 'triplet')
        temperature: Temperature parameter for temperature-scaled losses
        margin: Margin for triplet loss
        reduction: Reduction method
    """
    
    def __init__(
        self,
        loss_type: str = "infonce",
        temperature: float = 0.07,
        margin: float = 1.0,
        reduction: str = 'mean',
        block_size: int = 1,
        neg_sampling: str = "random",
        neg_k: int = 0,
        random_per_row: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.block_size = block_size
        
        if self.loss_type == "infonce":
            self.loss_fn = InfoNCELoss(
                temperature=temperature,
                reduction=reduction,
                block_size=block_size,
                neg_sampling=neg_sampling,
                neg_k=neg_k,
                random_per_row=random_per_row,
            )
        elif self.loss_type == "infonce2d":
            self.loss_fn = InfoNCELoss(
                temperature=temperature,
                reduction=reduction,
                is_2d=True,
                block_size=block_size,
                neg_sampling=neg_sampling,
                neg_k=neg_k,
                random_per_row=random_per_row,
            )
        elif self.loss_type == "ntxent":
            self.loss_fn = NTXentLoss(temperature=temperature, reduction=reduction)
        elif self.loss_type == "triplet":
            self.loss_fn = TripletLoss(margin=margin, reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, embeddings_0: torch.Tensor, embeddings_1: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the selected loss function."""
        return self.loss_fn(embeddings_0, embeddings_1, masks=masks)


def create_loss(
    loss_type: str = "infonce",
    temperature: float = 0.07,
    margin: float = 1.0,
    reduction: str = 'mean',
    block_size: int = 1,
    neg_sampling: str = "random",
    neg_k: int = 0,
    random_per_row: bool = True,
) -> nn.Module:
    """
    Factory function to create contrastive loss functions.
    
    Args:
        loss_type: Type of loss ('infonce', 'infonce2d', 'ntxent', 'triplet')
        temperature: Temperature parameter for temperature-scaled losses
        margin: Margin for triplet loss
        reduction: Reduction method
        
    Returns:
        Configured loss function
    """
    return ContrastiveLoss(
        loss_type=loss_type,
        temperature=temperature,
        margin=margin,
        reduction=reduction,
        block_size=block_size,
        neg_sampling=neg_sampling,
        neg_k=neg_k,
        random_per_row=random_per_row,
    )
