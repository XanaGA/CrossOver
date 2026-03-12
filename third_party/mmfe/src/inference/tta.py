import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from mmfe_utils.dino_utils import get_last_feature_dino
from mmfe_utils.tensor_utils import torch_erode
from mmfe_utils.aligment import apply_affine_2d_map, apply_affine_2d_points, estimate_affine_matrix_multiple, inverse_affine_matrix, params_to_affine_matrix
from mmfe_utils.tensor_utils import norm_tensor_to_pil


def vote_for_best_augmentation(
    embeddings_0: torch.Tensor,
    aug_embeddings_wrapped: torch.Tensor,
    num_samples: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vote for the best augmentation for each element in the batch.

    Args:
        embeddings_0: Original embeddings (B, C, H, W)
        aug_embeddings_wrapped: Augmented embeddings (F, B, C, H, W)
        num_samples: Number of points to sample for voting

    Returns:
        best_embeddings: Best augmented embeddings (B, C, H, W)
        best_aug_idx: Indices of the best augmentations (B,)
    """
    # Shapes
    num_augs, B, C, H, W = aug_embeddings_wrapped.shape
    N = H * W

    # Flatten spatial dims
    emb0 = embeddings_0.view(B, C, N)                  # (B, C, N)
    aug_emb = aug_embeddings_wrapped.view(num_augs, B, C, N)  # (F, B, C, N)

    # Normalize (cosine similarity)
    emb0 = torch.nn.functional.normalize(emb0, dim=1)      # (B, C, N)
    aug_emb = torch.nn.functional.normalize(aug_emb, dim=2)  # (F, B, C, N)

    # Sample spatial positions (M)
    if num_samples < N:
        sample_idx = torch.randint(0, N, (num_samples,), device=embeddings_0.device)
    else:
        sample_idx = torch.arange(N, device=embeddings_0.device)
    M = sample_idx.shape[0]

    emb0_sampled = emb0[:, :, sample_idx]  # (B, C, M)

    # Prepare for similarity: emb0_sampled -> (B, M, C), aug_emb -> (F, B, N, C)
    emb0_sampled_t = emb0_sampled.permute(0, 2, 1)   # (B, M, C)
    aug_emb_t = aug_emb.permute(0, 1, 3, 2)         # (F, B, N, C)

    # Compute similarity: (F, B, M, N)
    sim = torch.einsum("bmc,fbnc->fbmn", emb0_sampled_t, aug_emb_t)

    # For each (B, M) sample we want the single best (f, n) pair:
    # Permute to (B, M, F, N) and flatten the last two dims to (B, M, F*N)
    sim_b_m_f_n = sim.permute(1, 2, 0, 3)            # (B, M, F, N)
    sim_flat = sim_b_m_f_n.reshape(B, M, num_augs * N)  # (B, M, F*N)

    # argmax over flattened (F*N) -> gives index k in [0, F*N) for each (B, M)
    k = sim_flat.argmax(dim=2)                       # (B, M), dtype=torch.long

    # decode augmentation index: f = k // N
    best_aug_for_point = k // N                      # (B, M)

    # Count votes per augmentation for each batch element
    # one_hot -> (B, M, F), sum over M -> (B, F)
    one_hot = torch.nn.functional.one_hot(best_aug_for_point, num_classes=num_augs)  # (B, M, F)
    votes_b_f = one_hot.sum(dim=1)                   # (B, F)

    # Best augmentation index per batch
    best_aug_idx = votes_b_f.argmax(dim=1)           # (B,)

    # Gather the selected augmented embeddings: aug_embeddings_wrapped has shape (F, B, C, H, W)
    b_idx = torch.arange(B, device=aug_embeddings_wrapped.device)
    best_embeddings = aug_embeddings_wrapped[best_aug_idx, b_idx]  # (B, C, H, W)

    return best_embeddings, best_aug_idx, votes_b_f


def extract_selected_params(
    best_aug_idx: torch.Tensor,
    all_aug_params: list[dict],
    og_valid_masks: torch.Tensor,
    device: torch.device = None,
    numpy: bool = False
) -> dict:
    """
    Extract selected augmentation parameters based on best augmentation indices.
    
    Args:
        best_aug_idx: Indices of the best augmentations (B,)
        all_aug_params: List of augmentation parameter dictionaries
        og_valid_masks: Original valid masks (B, C, H, W)
        device: Device to use for tensors
    
    Returns:
        selected_params: Dictionary containing selected parameters with keys:
            - angle: torch.Tensor (B,)
            - translate: torch.Tensor (B, 2)
            - scale: torch.Tensor (B,)
            - shear: torch.Tensor (B,)
            - image_size: tuple (H, W)
            - valid_mask: torch.Tensor (B, C, H, W)
    """
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

        if og_valid_masks is not None:
            valid_mask = TF.affine(
                og_valid_masks[i],
                all_aug_params[best_idx]["angle"],
                all_aug_params[best_idx]["translate"],
                all_aug_params[best_idx]["scale"],
                all_aug_params[best_idx]["shear"],
                interpolation=TF.InterpolationMode.NEAREST,
            )
            valid_mask = torch_erode(valid_mask, kernel_size=3, iterations=1)
            selected_params["valid_mask"].append(valid_mask)

    if numpy:
        selected_params["angle"] = np.array(selected_params["angle"])
        selected_params["translate"] = np.array(selected_params["translate"])
        selected_params["scale"] = np.array(selected_params["scale"])
        selected_params["shear"] = np.array(selected_params["shear"])
        selected_params["image_size"] = selected_params["image_size"][0]
        if og_valid_masks is not None:
            selected_params["valid_mask"] = torch.stack(selected_params["valid_mask"]).cpu().numpy()
    else:
        selected_params["angle"] = torch.tensor(selected_params["angle"], device=device)
        selected_params["translate"] = torch.tensor(selected_params["translate"], device=device)
        selected_params["scale"] = torch.tensor(selected_params["scale"], device=device)
        selected_params["shear"] = torch.tensor(selected_params["shear"], device=device)
        selected_params["image_size"] = selected_params["image_size"][0]
        if og_valid_masks is not None:
            selected_params["valid_mask"] = torch.stack(selected_params["valid_mask"])

    return selected_params


def run_tta(embeddings_0: torch.Tensor, image_m1_noise: torch.Tensor, 
            og_valid_masks: torch.Tensor, model: torch.nn.Module, 
            filler: torch.Tensor = None, n_augs: int = 8, model_name: str = None, 
            upsampler: torch.nn.Module = None, upsampler_output_size: tuple = None,
            return_all: bool = False):
    """
    Run TTA for the given batch.
    """
    device = image_m1_noise.device

    aug_step = 360 / n_augs
    aug_embeddings = []
    all_aug_params = []
    if filler is None:
        filler = (1-torch.tensor([0.485, 0.456, 0.406]))/torch.tensor([0.229, 0.224, 0.225])
    filler = filler.view(-1, 1, 1).to(device).unsqueeze(0)

    if model_name is not None and model_name.startswith("dino"):
        embeddings_1_noisy_aug = get_last_feature_dino(model, image_m1_noise.to(device), model_name)
        if upsampler is not None and upsampler.mode == "anyup":
            embeddings_1_noisy_aug = upsampler(embeddings_1_noisy_aug, output_size=upsampler_output_size, original_images=image_m1_noise.to(device))
        elif upsampler is not None and upsampler.mode == "bilinear":
            embeddings_1_noisy_aug = upsampler(embeddings_1_noisy_aug, output_size=upsampler_output_size)
    else:
        embeddings_1_noisy_aug = model.get_embeddings(image_m1_noise)
    aug_embeddings.append(embeddings_1_noisy_aug)
    all_aug_params.append({"angle": 0, "translate": [0., 0.], "scale": 1.0, 
                            "shear": 0.0, "image_size": image_m1_noise.shape[-2:]})

    for i in range(1, n_augs):
        # Rotate modality_1_noise with angle i*20 degrees
        angle = i * aug_step
        # print(f"Augmentation {i} of {n_augs}: Angle {angle}")
        mask = og_valid_masks.clone()
        mask = TF.affine(mask, angle, [0., 0.], 1.0, 0.0, interpolation=TF.InterpolationMode.NEAREST).to(device)
        
        modality_1_noise_rot = TF.affine(image_m1_noise, angle, [0., 0.], 1.0, 0.0, interpolation=TF.InterpolationMode.BILINEAR)
        modality_1_noise_rot = torch.where(~mask.repeat(1, modality_1_noise_rot.shape[1], 1, 1).bool(), filler, modality_1_noise_rot)
        if model_name is not None and model_name.startswith("dino"):
            embeddings_1_noisy_aug = get_last_feature_dino(model, modality_1_noise_rot.to(device), model_name)
            if upsampler is not None and upsampler.mode == "anyup":
                embeddings_1_noisy_aug = upsampler(embeddings_1_noisy_aug, output_size=upsampler_output_size, original_images=modality_1_noise_rot.to(device))
            elif upsampler is not None and upsampler.mode == "bilinear":
                embeddings_1_noisy_aug = upsampler(embeddings_1_noisy_aug, output_size=upsampler_output_size)
        else:
            embeddings_1_noisy_aug = model.get_embeddings(modality_1_noise_rot)

        # Create parameters for this augmentation
        params_aug = {}
        params_aug["angle"] = angle
        params_aug["translate"] = [0., 0.]
        params_aug["scale"] = 1.0
        params_aug["shear"] = 0.0
        params_aug["image_size"] = image_m1_noise.shape[-2:]

        aug_embeddings.append(embeddings_1_noisy_aug)
        all_aug_params.append(params_aug)

    # Vote for the best augmentation
    aug_embeddings = torch.stack(aug_embeddings)  # (F, B, C, H, W)

    if return_all:
        return embeddings_0, aug_embeddings.permute(1, 0, 2, 3, 4), all_aug_params

    else:
        best_embedding_1_aug, best_aug_idx, votes_per_aug = vote_for_best_augmentation(embeddings_0, aug_embeddings)

        selected_params = extract_selected_params(best_aug_idx, all_aug_params, og_valid_masks, device)

        return best_embedding_1_aug, selected_params, best_aug_idx, votes_per_aug



def run_roma_tta(image0: torch.Tensor, image1: torch.Tensor, model, device: str, 
                n_augmentations: int = 8, filter_by_certainty: bool = True, version: str = "roma_v2"):
    """
    Run test-time augmentation for RoMa by rotating image0 multiple times and selecting matches with highest certainty.
    
    Args:
        image0: First image tensor (C, H, W)
        image1: Second image tensor (C, H, W)
        model: RegressionMatcher model instance
        device: Device to run on
        n_augmentations: Number of augmentations (rotations) to apply
        filter_by_certainty: If True, filter and return top matches by certainty. 
                            If False, return all matches organized by augmentation (N_AUGS, N_pts, 2)
    
    Returns:
        If filter_by_certainty=True:
            idx0: Corresponding points in image0 (N, 2)
            idx1: Corresponding points in image1 (N, 2)
        If filter_by_certainty=False:
            idx0: Corresponding points in image0 (N_AUGS, N_pts, 2)
            idx1: Corresponding points in image1 (N_AUGS, N_pts, 2)
    """
    aug_step = 360.0 / n_augmentations
    all_idx0 = []
    all_idx1 = []
    all_certainty = []
    
    # Process each augmentation
    for i in range(n_augmentations):
        angle = i * aug_step
        
        # Rotate image0
        if i == 0:
            # No rotation for first augmentation
            image0_rot = image0
        else:
            image0_rot = TF.affine(
                image0.unsqueeze(0), 
                angle, 
                [0., 0.], 
                1.0, 
                0.0, 
                interpolation=TF.InterpolationMode.BILINEAR
            ).squeeze(0)
        
        with torch.no_grad():
            if version == "roma_v2":
                preds = model.match(norm_tensor_to_pil(image0_rot), norm_tensor_to_pil(image1))
                matches, overlaps, precision_AtoB, precision_BtoA = model.sample(preds, 5000)

                # Convert to pixel coordinates (RoMaV2 produces matches in [-1,1]x[-1,1])
                idx0_aug, idx1_aug = model.to_pixel_coordinates(matches, image0_rot.shape[1], image0_rot.shape[2], image1.shape[1], image1.shape[2])
                certainty = precision_AtoB
            
            elif version == "roma_v1":
                preds, certainty = model.match(norm_tensor_to_pil(image0_rot), norm_tensor_to_pil(image1))
                matches, certainty = model.sample(preds, certainty)

                # Convert to pixel coordinates (RoMaV2 produces matches in [-1,1]x[-1,1])
                idx0_aug, idx1_aug = model.to_pixel_coordinates(matches, image0_rot.shape[1], image0_rot.shape[2], image1.shape[1], image1.shape[2])

        # Rotate idx0 back to original image0 coordinate system
        if i > 0:
            # Create inverse rotation affine matrix (rotate by -angle to get back to original)
            center = np.array([image0.shape[1] // 2, image0.shape[2] // 2])
            # theta_rad = -np.deg2rad(angle)  # Negative angle to rotate back
            # cos_theta = np.cos(theta_rad)
            # sin_theta = np.sin(theta_rad)
            
            # # Create rotation matrix (2x2) and convert to affine (2x3)
            # rot_matrix = np.array([[cos_theta, -sin_theta],
            #                        [sin_theta, cos_theta]])
            # rot_affine = np.hstack([rot_matrix, np.array([[0], [0]])])
            
            # # Use existing utility to apply inverse rotation
            # idx0_aug = apply_affine_2d_points(idx0_aug, rot_affine, center=center, device=device)

            params = {
                "angle": torch.tensor(angle),
                "scale": torch.tensor(1.0),
                "translate_x": torch.tensor(0.0),
                "translate_y": torch.tensor(0.0),
            }
            aff = params_to_affine_matrix(params)
            inv_aff = inverse_affine_matrix(aff)

            idx0_aug = apply_affine_2d_points(
                idx0_aug, inv_aff, center=center
            )
                
        # Store matches and certainty
        all_idx0.append(idx0_aug)
        all_idx1.append(idx1_aug)
        all_certainty.append(certainty)
    
    if filter_by_certainty:
        # Concatenate all matches
        all_idx0 = torch.cat(all_idx0, dim=0)  # (N_total, 2)
        all_idx1 = torch.cat(all_idx1, dim=0)   # (N_total, 2)
        all_certainty = torch.cat(all_certainty, dim=0)  # (N_total,)
        
        # Select matches with highest certainty
        # Sort by certainty in descending order
        sorted_indices = torch.argsort(all_certainty, descending=True)
        
        # Take top matches (keep top 1000 or all if fewer)
        top_k = min(1000, len(all_certainty))
        top_indices = sorted_indices[:top_k]
        
        idx0 = all_idx0[top_indices]
        idx1 = all_idx1[top_indices]
        
        return idx0, idx1
    else:
        # Return all matches organized by augmentation
        # Stack to get (N_AUGS, N_pts, 2) shape
        # Note: Different augmentations may have different numbers of matches
        # We'll pad to the maximum number of matches
        max_pts = max([idx.shape[0] for idx in all_idx0])
        
        # Pad all to same size
        idx0_padded = []
        idx1_padded = []
        for idx0_aug, idx1_aug in zip(all_idx0, all_idx1):
            n_pts = idx0_aug.shape[0]
            if n_pts < max_pts:
                # Pad with NaN or zeros (we'll use NaN to mark invalid points)
                padding = torch.full((max_pts - n_pts, 2), float('nan'), device=device, dtype=idx0_aug.dtype)
                idx0_padded.append(torch.cat([idx0_aug, padding], dim=0))
                idx1_padded.append(torch.cat([idx1_aug, padding], dim=0))
            else:
                idx0_padded.append(idx0_aug)
                idx1_padded.append(idx1_aug)
        
        idx0 = torch.stack(idx0_padded)  # (N_AUGS, N_pts, 2)
        idx1 = torch.stack(idx1_padded)  # (N_AUGS, N_pts, 2)
        
        return idx0, idx1
