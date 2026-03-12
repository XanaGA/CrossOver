#!/usr/bin/env python3
"""
Analyze alignment between two modality embeddings across the entire validation dataset.

This script mirrors `evaluate_aligment.py` but adds:
- Visualizations for *all* alignments whose accuracy at a configurable pixel
  threshold is below a configurable minimum accuracy.
- Aggregated statistics of which modality pairs are involved in these bad
  alignments.
- Statistics on how often Test Time Augmentation (TTA) selects the correct
  augmentation.
"""

import os
import sys
import gc
import matplotlib
from mmfe_utils.dino_utils import get_last_feature_dino, load_dino
from mmfe_utils.tensor_utils import norm_tensor_to_pil
from roma.roma_pl_module import RoMaFineTuner

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cv2
from hydra.utils import to_absolute_path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torch.nn as nn

from inference.tta import run_tta, run_roma_tta
from mmfe_utils.tensor_utils import torch_erode

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dataloading.unified_dataset import UnifiedDataset  # noqa: E402
from training.lightning_module import ContrastiveLearningModule  # noqa: E402
from mmfe_utils.aligment import (  # noqa: E402
    compose_affine_matrices,
    find_nn,
    inverse_affine_matrix,
    params_to_affine_matrix,
    apply_affine_2d_map,
    apply_affine_2d_points,
    estimate_affine_matrix,
    estimate_affine_matrix_multiple,
    evaluate_corner_alignment,
)
from omegaconf import DictConfig, OmegaConf  # noqa: E402
import hydra  # noqa: E402
from tqdm import tqdm  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

try:
    from romav2 import RoMaV2  # type: ignore
except ImportError:
    print("RoMaV2 not found")

try:
    from romatch import roma_indoor  # type: ignore
    from romatch.models.matcher import RegressionMatcher  # type: ignore
except ImportError:
    print("RoMaV1 not found")

from mmfe_utils.data_utils import create_val_dataset  # noqa: E402


class MyUpsampler(torch.nn.Module):
    def __init__(self, mode: str = "bilinear", neural_upsampler: torch.nn.Module = None):
        super(MyUpsampler, self).__init__()
        self.mode = mode
        self.neural_upsampler = neural_upsampler

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple,
        original_images: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.mode == "bilinear":
            return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        elif self.mode == "nearest":
            return F.interpolate(x, size=output_size, mode="nearest", align_corners=False)
        elif self.mode == "anyup":
            assert self.neural_upsampler is not None, "neural_upsampler must be provided for anyup mode"
            return self.neural_upsampler(original_images, x, output_size)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


class RoMaBackboneWrapper(nn.Module):
    """
    Wraps a custom backbone producing 32-D features and expands them
    into the 1024-D × 2 features required by RoMa.
    """

    def __init__(
        self,
        backbone,
        in_dim: int = 32,
        out_dim: int = 1024,
        projection: str = "conv",  # "conv", "bilinear", or nn.Module
    ):
        super().__init__()
        self.backbone = backbone
        self.in_dim = in_dim
        self.out_dim = out_dim

        if isinstance(projection, nn.Module):
            self.proj = projection
            self.use_custom = True
        elif projection == "conv":
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
            self.use_custom = False
        elif projection == "bilinear":
            self.proj = None
            self.use_bilinear = True
        else:
            raise ValueError(f"Unknown projection type: {projection}")

        if projection != "bilinear":
            self.use_bilinear = False

    def _bilinear_expand(self, f: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            return f

        reps = self.out_dim // self.in_dim
        out = f.repeat(1, reps, 1, 1)

        if out.shape[1] < self.out_dim:
            extra = self.out_dim - out.shape[1]
            out = torch.cat([out, f[:, :extra]], dim=1)

        return out

    def forward(self, x: torch.Tensor):
        # Backbone must return (B, in_dim, H/16, W/16)
        f = self.backbone(x)

        if self.use_bilinear:
            f1 = self._bilinear_expand(f).permute(0, 2, 3, 1)
            f2 = self._bilinear_expand(f).permute(0, 2, 3, 1)
        else:
            f1 = self.proj(f).permute(0, 2, 3, 1)
            f2 = self.proj(f).permute(0, 2, 3, 1)

        return [f1, f2]


def denormalize_tensor(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def angle_from_rotation(R: np.ndarray) -> float:
    theta = np.arctan2(R[1, 0], R[0, 0])
    return float(np.degrees(theta))


def get_accuracy_for_threshold(sample_result: dict, pixel_threshold: float):
    """
    Helper to read accuracy@{threshold} handling both integer and float keys.
    """
    int_key = f"accuracy@{int(pixel_threshold)}"
    float_key = f"accuracy@{float(pixel_threshold)}"

    acc = sample_result.get(int_key, None)
    if acc is None or (isinstance(acc, float) and np.isnan(acc)):
        acc = sample_result.get(float_key, None)
    return acc


def save_bad_alignment_visualizations_for_batch(
    batch: dict,
    batch_results: list,
    batch_idx: int,
    cfg: DictConfig,
    pixel_threshold: float,
    min_accuracy: float,
    output_dir: str,
):
    """
    Save a 1x3 grid image for each alignment in the batch whose accuracy at the
    given pixel_threshold is below min_accuracy.
    """
    os.makedirs(output_dir, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    norm_filler = (1 - mean) / std

    corner_labels = ["TL", "TR", "BL", "BR"]

    for res in batch_results:
        acc = get_accuracy_for_threshold(res, pixel_threshold)
        if acc is None or np.isnan(acc) or acc >= min_accuracy:
            continue

        i = res["batch_idx"]
        img0 = denormalize_tensor(batch["modality_0"][i].cpu(), mean, std).permute(1, 2, 0).numpy()

        # Decide which modality 1 to show (mirror evaluation logic)
        if cfg.transforms.tf_difficulty is not None and not cfg.transforms.rotate_after and "modality_1_noise" in batch:
            img1_disp_tensor = batch["modality_1_noise"][i].cpu()
        else:
            img1_orig = batch["modality_1"][i].cpu()
            actual_affine = np.array(res.get("actual_affine")) if res.get("actual_affine") is not None else None

            if res.get("rotate_after", False):
                if actual_affine is not None:
                    img1_disp_tensor = apply_affine_2d_map(img1_orig, actual_affine).cpu()
                    mask = apply_affine_2d_map(
                        torch.ones(1, img1_orig.shape[1], img1_orig.shape[2]),
                        actual_affine,
                    )
                    mask = torch_erode(mask, kernel_size=3)
                    black_threshold = 0.5
                    img1_disp_tensor = torch.where(
                        ~(mask.repeat(3, 1, 1) < black_threshold),
                        img1_disp_tensor,
                        norm_filler[..., None, None],
                    )
                else:
                    img1_disp_tensor = img1_orig
            else:
                if actual_affine is not None and cfg.transforms.tf_difficulty is None:
                    img1_disp_tensor = apply_affine_2d_map(img1_orig, actual_affine).cpu()
                    mask = apply_affine_2d_map(
                        torch.ones(1, img1_orig.shape[1], img1_orig.shape[2]),
                        actual_affine,
                    )
                    mask = torch_erode(mask, kernel_size=3)
                    black_threshold = 0.5
                    img1_disp_tensor = torch.where(
                        ~(mask.repeat(3, 1, 1) < black_threshold),
                        img1_disp_tensor,
                        norm_filler[..., None, None],
                    )
                else:
                    img1_disp_tensor = img1_orig

        img1 = denormalize_tensor(img1_disp_tensor, mean, std).permute(1, 2, 0).numpy()

        # Correct modality 1 by the estimated affine
        aff_est = np.array(res.get("aff_est")) if res.get("aff_est") is not None else None
        if aff_est is not None:
            corrected = apply_affine_2d_map(img1_disp_tensor, inverse_affine_matrix(aff_est))
            mask = apply_affine_2d_map(
                torch.ones(1, img1_disp_tensor.shape[1], img1_disp_tensor.shape[2]),
                inverse_affine_matrix(aff_est),
            )
            mask = torch_erode(mask, kernel_size=3)
            black_threshold = 0.5
            corrected = torch.where(
                ~(mask.repeat(3, 1, 1) < black_threshold),
                corrected,
                norm_filler[..., None, None],
            )
            corrected = denormalize_tensor(corrected, mean, std).permute(1, 2, 0).cpu().numpy()
        else:
            corrected = img1

        corners_original = np.array(res.get("corners_original", [])) if res.get("corners_original") is not None else None
        corners_gt = np.array(res.get("corners_gt", [])) if res.get("corners_gt") is not None else None
        corners_pred = np.array(res.get("corners_pred", [])) if res.get("corners_pred") is not None else None

        fig, axs = plt.subplots(1, 3, figsize=(12, 5))

        axs[0].imshow(img0)
        axs[0].set_title("Modality 0")
        axs[0].axis("off")
        if corners_original is not None and len(corners_original) > 0:
            for corner, label in zip(corners_original, corner_labels):
                axs[0].scatter(
                    corner[0],
                    corner[1],
                    color="red",
                    s=80,
                    marker="o",
                    edgecolors="white",
                    linewidth=2,
                    alpha=0.8,
                )
                axs[0].text(
                    corner[0] + 5,
                    corner[1] - 5,
                    f"{label}_GT",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.7),
                )

        axs[1].imshow(img1)
        axs[1].set_title("Modality 1 (as used)")
        axs[1].axis("off")
        if corners_gt is not None and len(corners_gt) > 0:
            for corner, label in zip(corners_gt, corner_labels):
                axs[1].scatter(
                    corner[0],
                    corner[1],
                    color="green",
                    s=80,
                    marker="o",
                    edgecolors="white",
                    linewidth=2,
                    alpha=0.8,
                )
                axs[1].text(
                    corner[0] + 5,
                    corner[1] - 5,
                    label,
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="green", alpha=0.7),
                )

        acc10 = get_accuracy_for_threshold(res, 10.0)
        acc10 = 0.0 if acc10 is None or np.isnan(acc10) else acc10
        axs[2].imshow(corrected)
        axs[2].set_title(f"Corrected | Acc@{int(pixel_threshold)}: {acc:.1f}% | Acc@10: {acc10:.1f}%")
        axs[2].axis("off")

        if (
            corners_gt is not None
            and corners_pred is not None
            and len(corners_gt) > 0
            and len(corners_pred) > 0
        ):
            actual_affine = np.array(res.get("actual_affine")) if res.get("actual_affine") is not None else None
            if actual_affine is not None:
                corrected_corners = apply_affine_2d_points(
                    corners_pred,
                    inverse_affine_matrix(actual_affine),
                    center=np.array([img1.shape[1] // 2, img1.shape[0] // 2]),
                )
            else:
                corrected_corners = corners_pred

            for gt_corner, pred_corner in zip(corners_original, corrected_corners):
                axs[2].plot(
                    [gt_corner[0], pred_corner[0]],
                    [gt_corner[1], pred_corner[1]],
                    "yellow",
                    linewidth=2,
                    alpha=0.6,
                )

            for corner, label in zip(corners_original, corner_labels):
                axs[2].scatter(
                    corner[0],
                    corner[1],
                    color="green",
                    s=80,
                    marker="o",
                    edgecolors="white",
                    linewidth=2,
                    alpha=0.8,
                )
                axs[2].text(
                    corner[0] + 5,
                    corner[1] - 5,
                    f"{label}_GT",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="green", alpha=0.7),
                )

        plt.tight_layout()

        global_sample_idx = res.get("global_sample_idx", i)
        m0_type = batch.get("m0_type", [None])[i] if "m0_type" in batch else None
        m1_type = batch.get("m1_type", [None])[i] if "m1_type" in batch else None
        modality_pair = f"{m0_type}->{m1_type}" if (m0_type is not None and m1_type is not None) else "unknown"

        fig_name = (
            f"bad_alignment_batch{batch_idx:04d}_sample{global_sample_idx:06d}_"
            f"{modality_pair}_th{int(pixel_threshold)}.png"
        )
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def process_batch(
    model,
    batch,
    true_affine,
    cfg: DictConfig,
    device,
    model_name: str = None,
    upsampler: torch.nn.Module = None,
    upsampler_output_size: tuple = (32, 32),
):
    """
    Process a single batch and return alignment metrics.

    Compared to `evaluate_aligment.process_batch`, this version also records
    TTA correctness information per sample when applicable.
    """
    image0, image1 = batch["modality_0"], batch["modality_1"]

    image0 = image0.to(device)
    image1 = image1.to(device)

    norm_filler = batch["norm_filler"].to(device)

    if cfg.transforms.tf_difficulty is not None and not cfg.transforms.rotate_after:
        image1 = batch["modality_1_noise"].to(device)
        mask = batch["noise_params"]["valid_mask"].to(device)
    elif not cfg.transforms.rotate_after:
        image1 = apply_affine_2d_map(image1, true_affine)
        mask = apply_affine_2d_map(torch.ones(1, image1.shape[1], image1.shape[2]), true_affine).to(device)
        mask = torch_erode(mask, kernel_size=3)
        black_threshold = 0.5
        image1 = torch.where(
            ~(mask.repeat(1, 3, 1, 1) < black_threshold),
            image1,
            norm_filler[0].view(1, 3, 1, 1),
        )

    tta_correct_per_sample = None
    tta_best_aug_idx = None
    tta_gt_aug = None

    if not isinstance(model, RoMaV2) and not isinstance(model, RoMaFineTuner) and not isinstance(model, RegressionMatcher):
        with torch.no_grad():
            if model_name.startswith("dino"):
                e0 = get_last_feature_dino(model, image0.to(device), model_name)
                if upsampler is not None and upsampler.mode == "anyup":
                    e0 = upsampler(e0, output_size=upsampler_output_size, original_images=image0.to(device))
                elif upsampler is not None and upsampler.mode == "bilinear":
                    e0 = upsampler(e0, output_size=upsampler_output_size)
            else:
                e0 = model.get_embeddings(image0)

            if cfg.eval.tta_n_augs > 1:
                if mask.shape[0] != e0.shape[0]:
                    mask = mask.repeat(e0.shape[0], 1, 1, 1)

                if cfg.eval.use_inliers:
                    e0, e1, all_aug_params = run_tta(
                        e0,
                        image1,
                        mask,
                        model,
                        n_augs=cfg.eval.tta_n_augs,
                        model_name=model_name,
                        upsampler=upsampler,
                        upsampler_output_size=upsampler_output_size,
                        return_all=True,
                    )
                else:
                    best_embedding_1_aug, selected_params, best_aug_idx, votes_per_aug = run_tta(
                        e0,
                        image1,
                        mask,
                        model,
                        n_augs=cfg.eval.tta_n_augs,
                        model_name=model_name,
                        upsampler=upsampler,
                        upsampler_output_size=upsampler_output_size,
                    )
                    e1 = best_embedding_1_aug

                    aug_step = 360 / cfg.eval.tta_n_augs
                    gt_rot = (
                        batch["noise_params"]["angle"].to(device)
                        if "noise_params" in batch
                        else torch.tensor([cfg.transforms.angle] * e0.shape[0]).to(device)
                    )
                    gt_aug = (
                        (cfg.eval.tta_n_augs + (360 - gt_rot + aug_step / 2) // aug_step)
                        % cfg.eval.tta_n_augs
                    ).to(torch.int32)

                    # Track TTA correctness per sample
                    tta_best_aug_idx = best_aug_idx.to(gt_aug.device)
                    tta_gt_aug = gt_aug
                    tta_correct_per_sample = (tta_best_aug_idx == tta_gt_aug).to(torch.int32)
            else:
                if model_name.startswith("dino"):
                    e1 = get_last_feature_dino(model, image1.to(device), model_name)
                    e1 = upsampler(image1.to(device), e1, output_size=upsampler_output_size)
                else:
                    e1 = model.get_embeddings(image1)

    batch_size = image0.shape[0]
    batch_results = []

    for i in range(batch_size):
        try:
            if cfg.transforms.tf_difficulty is not None and "noise_params" in batch:
                actual_angle = float(batch["noise_params"]["angle"][i].cpu().numpy())
                actual_tx = float(batch["noise_params"]["translate"][1][i].cpu().numpy())
                actual_ty = float(batch["noise_params"]["translate"][0][i].cpu().numpy())
                actual_scale = float(batch["noise_params"]["scale"][i].cpu().numpy())

                theta_actual = np.deg2rad(actual_angle)
                R_actual = np.array(
                    [
                        [np.cos(theta_actual), -np.sin(theta_actual)],
                        [np.sin(theta_actual), np.cos(theta_actual)],
                    ]
                )
                S_actual = np.array([[actual_scale, 0], [0, actual_scale]])
                t_actual = np.array([actual_tx, actual_ty])
                A_actual = R_actual @ S_actual
                actual_affine = np.array(
                    [
                        [A_actual[0, 0], A_actual[0, 1], t_actual[1] / (cfg.data.image_size[1] / 2)],
                        [A_actual[1, 0], A_actual[1, 1], t_actual[0] / (cfg.data.image_size[0] / 2)],
                    ]
                )
            else:
                actual_angle = cfg.transforms.angle
                actual_tx = cfg.transforms.tx
                actual_ty = cfg.transforms.ty
                actual_scale = cfg.transforms.scale
                actual_affine = true_affine

            if isinstance(model, RoMaV2) or isinstance(model, RoMaFineTuner) or isinstance(model, RegressionMatcher):
                version = "roma_v2" if isinstance(model, RoMaV2) else "roma_v1"
                filter_by_certainty = cfg.eval.get("filter_by_certainty", True)

                if cfg.eval.tta_n_augs > 1:
                    idx0, idx1 = run_roma_tta(
                        image0[i],
                        image1[i],
                        model,
                        device,
                        n_augmentations=cfg.eval.tta_n_augs,
                        filter_by_certainty=filter_by_certainty,
                        version=version,
                    )
                else:
                    with torch.no_grad():
                        if version == "roma_v2":
                            preds = model.match(norm_tensor_to_pil(image0[i]), norm_tensor_to_pil(image1[i]))
                            matches, overlaps, precision_AtoB, precision_BtoA = model.sample(preds)
                            idx0, idx1 = model.to_pixel_coordinates(
                                matches,
                                image0[i].shape[1],
                                image0[i].shape[2],
                                image1[i].shape[1],
                                image1[i].shape[2],
                            )
                        elif version == "roma_v1":
                            preds, certainty = model.match(norm_tensor_to_pil(image0[i]), norm_tensor_to_pil(image1[i]))
                            matches, certainty = model.sample(preds, certainty)
                            idx0, idx1 = model.to_pixel_coordinates(
                                matches,
                                image0[i].shape[1],
                                image0[i].shape[2],
                                image1[i].shape[1],
                                image1[i].shape[2],
                            )

                H, W = image0[i].shape[1], image0[i].shape[2]
                center = np.array([H // 2, W // 2])
                has_multiple_augs = isinstance(idx0, torch.Tensor) and idx0.dim() == 3

                if cfg.eval.use_gt_correspondances:
                    if has_multiple_augs and not filter_by_certainty:
                        idx1_list = []
                        for aug_idx in range(idx0.shape[0]):
                            idx0_aug = idx0[aug_idx].cpu()
                            valid_mask = ~(torch.isnan(idx0_aug).any(dim=1))
                            if valid_mask.sum() > 0:
                                idx0_valid = idx0_aug[valid_mask]
                                idx1_aug = apply_affine_2d_points(idx0_valid, actual_affine, center=center)
                                idx1_full = torch.full(
                                    (idx0_aug.shape[0], 2),
                                    float("nan"),
                                    device=idx1_aug.device,
                                    dtype=idx1_aug.dtype,
                                )
                                idx1_full[valid_mask] = idx1_aug
                                idx1_list.append(idx1_full)
                            else:
                                idx1_list.append(
                                    torch.full(
                                        (idx0_aug.shape[0], 2),
                                        float("nan"),
                                        device=idx0_aug.device,
                                        dtype=idx0_aug.dtype,
                                    )
                                )
                        idx1 = torch.stack(idx1_list)
                    else:
                        if has_multiple_augs:
                            raise Exception("ERROR: Multiple augmentations found but filter_by_certainty is True")
                        idx1 = apply_affine_2d_points(idx0.cpu(), actual_affine, center=center)

                if has_multiple_augs and not filter_by_certainty:
                    aff_est, best_aug_idx, n_inliers = estimate_affine_matrix_multiple(
                        idx0.cpu(),
                        idx1.cpu(),
                        center=center,
                        method=cfg.eval.method,
                    )
                else:
                    if has_multiple_augs:
                        idx0 = idx0[0]
                        idx1 = idx1[0]
                    aff_est = estimate_affine_matrix(
                        idx0.cpu(),
                        idx1.cpu(),
                        center=center,
                        method=cfg.eval.method,
                    )
            else:
                if cfg.transforms.rotate_after:
                    e1_rot = apply_affine_2d_map(e1[i], actual_affine)
                    if "noise_params" in batch:
                        mask = batch["noise_params"]["valid_mask"][i].to(device)
                    else:
                        mask = apply_affine_2d_map(torch.ones(1, e1.shape[2], e1.shape[3]), actual_affine)
                        mask = torch_erode(mask, kernel_size=3)
                else:
                    e1_rot = e1[i]
                    mask = None

                if cfg.eval.use_inliers:
                    H, W = e1_rot.shape[1], e1_rot.shape[2]
                    e_center = np.array([H // 2, W // 2])
                    e0_rot = e0[i]
                    e1_rot = e1_rot
                    idx0_list, idx1_list = [], []
                    for aug_idx in range(e1_rot.shape[0]):
                        i0, _, i1, _ = find_nn(e0_rot, e1_rot[aug_idx], mask=mask, top_k=100)

                        params = {
                            "angle": torch.tensor(0.0),
                            "scale": torch.tensor(1.0),
                            "translate_x": torch.tensor(0.0),
                            "translate_y": torch.tensor(0.0),
                        }
                        aff = params_to_affine_matrix(params)
                        inv_aff = inverse_affine_matrix(aff)

                        i1 = apply_affine_2d_points(i1, inv_aff, center=e_center)

                        idx0_list.append(i0)
                        idx1_list.append(i1)

                    idx0 = torch.stack(idx0_list)
                    idx1 = torch.stack(idx1_list)
                    aff_est, best_aug_idx, n_inliers = estimate_affine_matrix_multiple(
                        idx0.cpu(),
                        idx1.cpu(),
                        center=e_center,
                        method=cfg.eval.method,
                    )
                else:
                    if upsampler is not None and upsampler.mode == "anyup":
                        e0_rot = upsampler(
                            e0[i].unsqueeze(0),
                            output_size=upsampler_output_size,
                            original_images=image0[i].unsqueeze(0).to(device),
                        ).squeeze(0)
                        e1_rot_tmp = upsampler(
                            e1_rot.unsqueeze(0),
                            output_size=upsampler_output_size,
                            original_images=image1[i].unsqueeze(0).to(device),
                        ).squeeze(0)
                    elif upsampler is not None and upsampler.mode == "bilinear":
                        e0_rot = upsampler(
                            e0[i].unsqueeze(0),
                            output_size=upsampler_output_size,
                        ).squeeze(0)
                        e1_rot_tmp = upsampler(
                            e1_rot.unsqueeze(0),
                            output_size=upsampler_output_size,
                        ).squeeze(0)
                    else:
                        e0_rot = e0[i]
                        e1_rot_tmp = e1_rot

                    idx0, _, idx1, _ = find_nn(e0_rot, e1_rot_tmp, mask=mask, top_k=100)

                    H, W = e1_rot_tmp.shape[1], e1_rot_tmp.shape[2]
                    center = np.array([H // 2, W // 2])

                    if cfg.eval.use_gt_correspondances:
                        idx1 = apply_affine_2d_points(idx0.cpu(), actual_affine, center=center)

                    aff_est = estimate_affine_matrix(
                        idx0.cpu(),
                        idx1.cpu(),
                        center=center,
                        method=cfg.eval.method,
                    )

            if (
                cfg.eval.tta_n_augs > 1
                and not isinstance(model, RoMaV2)
                and not isinstance(model, RoMaFineTuner)
                and not isinstance(model, RegressionMatcher)
                and not cfg.eval.use_gt_correspondances
                and not cfg.eval.use_inliers
            ):
                params_i = {
                    "angle": selected_params["angle"][i].cpu().numpy(),
                    "scale": selected_params["scale"][i].cpu().numpy(),
                    "translate_x": selected_params["translate"][i][1].cpu().numpy(),
                    "translate_y": selected_params["translate"][i][0].cpu().numpy(),
                }
                aff_correcting_aug = params_to_affine_matrix(params_i)
                aff_est = compose_affine_matrices(inverse_affine_matrix(aff_correcting_aug), aff_est).cpu().numpy()

            theta_est = angle_from_rotation(aff_est[:2, :2])

            eval_results = evaluate_corner_alignment(
                actual_affine,
                aff_est,
                img_shape=tuple(cfg.data.image_size),
                threshold=cfg.eval.threshold,
            )

            tta_correct = None
            tta_best = None
            tta_gt = None
            if tta_correct_per_sample is not None:
                tta_correct = int(tta_correct_per_sample[i].item())
                tta_best = int(tta_best_aug_idx[i].item())
                tta_gt = int(tta_gt_aug[i].item())

            sample_result = {
                "batch_idx": i,
                "rotate_after": cfg.transforms.rotate_after,
                "tf_difficulty": cfg.transforms.tf_difficulty if cfg.transforms.tf_difficulty else "deterministic",
                "true_rotation": actual_angle,
                "estimated_rotation": theta_est,
                "rotation_error": abs(actual_angle - theta_est),
                "true_translation_x": actual_tx,
                "true_translation_y": actual_ty,
                "true_scale": actual_scale,
                "mean_distance": eval_results["mean_distance"],
                "max_distance": eval_results["max_distance"],
                "rms_error": eval_results["rms_error"],
                "median_error": eval_results["median_error"],
                "corner_distances": eval_results["distances"].tolist(),
                "corners_original": eval_results["corners_original"].tolist(),
                "corners_gt": eval_results["corners_gt"].tolist(),
                "corners_pred": eval_results["corners_pred"].tolist(),
                "threshold": cfg.eval.threshold,
                "aff_est": aff_est.tolist() if isinstance(aff_est, np.ndarray) else aff_est,
                "actual_affine": actual_affine.tolist() if isinstance(actual_affine, np.ndarray) else actual_affine,
                "tta_correct": tta_correct,
                "tta_best_aug_idx": tta_best,
                "tta_gt_aug": tta_gt,
            }

            for key, value in eval_results.items():
                if key.startswith("accuracy@"):
                    sample_result[key] = value

            batch_results.append(sample_result)

        except Exception as e:
            print(f"Error processing sample {i} in batch: {e}")
            sample_result = {
                "batch_idx": i,
                "rotate_after": cfg.transforms.rotate_after,
                "tf_difficulty": cfg.transforms.tf_difficulty if cfg.transforms.tf_difficulty else "deterministic",
                "true_rotation": np.nan,
                "estimated_rotation": np.nan,
                "rotation_error": np.nan,
                "true_translation_x": np.nan,
                "true_translation_y": np.nan,
                "true_scale": np.nan,
                "accuracy": np.nan,
                "mean_distance": np.nan,
                "max_distance": np.nan,
                "rms_error": np.nan,
                "median_rms_error": np.nan,
                "median_error": np.nan,
                "corner_distances": [np.nan, np.nan, np.nan, np.nan],
                "corners_original": None,
                "corners_gt": None,
                "corners_pred": None,
                "threshold": cfg.eval.threshold,
                "error": str(e),
                "aff_est": None,
                "actual_affine": None,
                "tta_correct": None,
                "tta_best_aug_idx": None,
                "tta_gt_aug": None,
            }
            batch_results.append(sample_result)

        torch.cuda.empty_cache()
        gc.collect()

    return batch_results


@hydra.main(config_path="../../configs", config_name="analyze_alignment", version_base="1.3")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    output_csv_abs = to_absolute_path(cfg.logging.output_csv)
    os.makedirs(os.path.dirname(output_csv_abs), exist_ok=True)

    # Analysis configuration (with sensible defaults if missing)
    pixel_threshold = 10.0
    min_accuracy = 90.0
    analysis_subdir = "analysis_viz"
    if "analysis" in cfg:
        pixel_threshold = float(cfg.analysis.get("pixel_threshold", pixel_threshold))
        min_accuracy = float(cfg.analysis.get("min_accuracy", min_accuracy))
        analysis_subdir = cfg.analysis.get("output_subdir", analysis_subdir)

    bad_alignments_dir = os.path.join(
        os.path.dirname(output_csv_abs),
        analysis_subdir,
        f"bad_alignments_th{int(pixel_threshold)}",
    )
    os.makedirs(bad_alignments_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(cfg, "runtime") and getattr(cfg.runtime, "device", None) in ("cuda", "cpu"):
        device = cfg.runtime.device

    print(f"Using device: {device}")

    upsampler = None
    upsampler_type = cfg.model.kwargs.get("upsampler_type", None)

    if upsampler_type is not None:
        if upsampler_type == "anyup":
            anyup = torch.hub.load("wimmerth/anyup", "anyup_multi_backbone", use_natten=False)
            upsampler = MyUpsampler(mode="anyup", neural_upsampler=anyup)
            upsampler.to(device)
            upsampler.eval()
        elif upsampler_type == "bilinear":
            upsampler = MyUpsampler(mode="bilinear")
            upsampler.to(device)

    if cfg.model.checkpoint == "roma_v2":
        dinov2_weights_path = os.getenv("DINO_ROMA_WEIGHTS_PATH", None)
        dinov2_weights_path = to_absolute_path(dinov2_weights_path) if dinov2_weights_path is not None else None
        _ = torch.load(dinov2_weights_path, map_location="cpu") if dinov2_weights_path is not None else None

        model = RoMaV2()

        if cfg.model.kwargs.get("roma_mmfe_checkpoint", None) is not None:
            backbone = ContrastiveLearningModule.load_from_checkpoint(
                checkpoint_path=to_absolute_path(cfg.model.kwargs.roma_mmfe_checkpoint),
                map_location=device,
                load_dino_weights=False,
                weights_only=False,
            )
            model.f = RoMaBackboneWrapper(backbone, in_dim=32, out_dim=1024, projection="bilinear")

        model.to(device)
        model.eval()
        model.compile()

    elif cfg.model.checkpoint == "roma_v1":
        model = roma_indoor(device=device)
        model.to(device)
        model.eval()

    elif cfg.model.checkpoint.startswith("dinov3"):
        dino_weights_path = os.getenv("DINOV3_WEIGHTS_PATH", None)
        dino_weights_path = to_absolute_path(dino_weights_path) if dino_weights_path is not None else None
        model = load_dino(cfg.model.checkpoint, load_dino_weights=True, dino_weights_path=dino_weights_path)
        model.to(device)
        model.eval()

    elif cfg.model.checkpoint.startswith("dinov2"):
        dino_weights_path = os.getenv("DINOV2_WEIGHTS_PATH", None)
        dino_weights_path = to_absolute_path(dino_weights_path) if dino_weights_path is not None else None
        model = load_dino(cfg.model.checkpoint, load_dino_weights=True, dino_weights_path=dino_weights_path)
        model.to(device)
        model.eval()

    elif "finetune" in cfg.model.checkpoint:
        model = RoMaFineTuner.load_from_checkpoint(
            checkpoint_path=to_absolute_path(cfg.model.checkpoint),
            map_location=device,
            weights_only=False,
            mmfe_roma_checkpoint_path=cfg.model.kwargs.get("roma_mmfe_checkpoint", None),
        )
        model.to(device)
        model.eval()
    else:
        if not cfg.model.checkpoint:
            raise ValueError("model.checkpoint must be provided (path to .ckpt)")
        model = ContrastiveLearningModule.load_from_checkpoint(
            checkpoint_path=to_absolute_path(cfg.model.checkpoint),
            map_location=device,
            load_dino_weights=False,
            weights_only=False,
        )
        model.to(device)
        model.eval()

    if (
        cfg.eval.batch_size > 4
        and cfg.model.checkpoint in ["dinov3_vitb16", "dinov2_vitb14"]
        and cfg.model.kwargs.get("use_dino_res", False)
    ):
        OmegaConf.set_struct(cfg.eval, False)
        new_bs = cfg.eval.batch_size // 4
        new_bs = new_bs if new_bs > 1 else 1
        cfg.eval.batch_size = new_bs

    if (
        cfg.model.checkpoint == "roma_v2"
        or cfg.model.checkpoint == "roma_v1"
        or (cfg.model.checkpoint in ["dinov3_vitb16", "dinov2_vitb14"] and cfg.model.kwargs.get("use_dino_res", False))
        or "finetune" in cfg.model.checkpoint
    ):
        new_H = 560
        new_W = 560
        OmegaConf.set_struct(cfg.data, False)
        cfg.data.image_size = [new_H, new_W]

    val_dataset = create_val_dataset(cfg)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if cfg.transforms.tf_difficulty is None:
        theta = np.deg2rad(cfg.transforms.angle)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        S = np.array([[cfg.transforms.scale, 0], [0, cfg.transforms.scale]])
        t = np.array([cfg.transforms.tx, cfg.transforms.ty])
        A = R @ S
        true_affine = np.array([[A[0, 0], A[0, 1], t[0]], [A[1, 0], A[1, 1], t[1]]])
        print(
            f"Applied deterministic affine transforms - Rotation: {cfg.transforms.angle:.2f}°, "
            f"Translation: ({cfg.transforms.tx:.1f}, {cfg.transforms.ty:.1f}), "
            f"Scale: {cfg.transforms.scale:.2f}"
        )
    else:
        true_affine = None
        print(f"Using random transformations with difficulty: {cfg.transforms.tf_difficulty}")

    print(f"Rotate after embeddings: {cfg.transforms.rotate_after}")
    print(f"Processing {len(val_loader)} batches with batch size {cfg.eval.batch_size}")
    if cfg.eval.max_batches:
        print(f"Limited to {cfg.eval.max_batches} batches")

    all_results = []
    bad_samples_records = []
    batch_count = 0

    for batch_idx, batch in enumerate(tqdm(val_loader)):
        if cfg.eval.max_batches and batch_idx >= cfg.eval.max_batches:
            break

        print(f"Processing batch {batch_idx + 1}/{len(val_loader)}")

        batch_results = process_batch(
            model,
            batch,
            true_affine,
            cfg,
            device,
            model_name=cfg.model.checkpoint,
            upsampler=upsampler,
            upsampler_output_size=tuple(cfg.model.kwargs.upsampler_output_size)
            if cfg.model.kwargs.get("upsampler_output_size", None) is not None
            else None,
        )

        for result in batch_results:
            result["global_batch_idx"] = batch_idx
            result["global_sample_idx"] = batch_idx * cfg.eval.batch_size + result["batch_idx"]

            i = result["batch_idx"]
            m0_type = batch.get("m0_type", [None])[i] if "m0_type" in batch else None
            m1_type = batch.get("m1_type", [None])[i] if "m1_type" in batch else None
            source_dataset = batch.get("source_dataset", [None])[i] if "source_dataset" in batch else None

            result["m0_type"] = m0_type
            result["m1_type"] = m1_type
            result["source_dataset"] = source_dataset
            modality_pair = None
            if m0_type is not None and m1_type is not None:
                modality_pair = f"{m0_type}->{m1_type}"
            result["modality_pair"] = modality_pair

            acc = get_accuracy_for_threshold(result, pixel_threshold)
            if acc is not None and not np.isnan(acc) and acc < min_accuracy:
                bad_samples_records.append(
                    {
                        "global_sample_idx": result["global_sample_idx"],
                        "source_dataset": source_dataset,
                        "m0_type": m0_type,
                        "m1_type": m1_type,
                        "modality_pair": modality_pair,
                        "accuracy": acc,
                        "pixel_threshold": pixel_threshold,
                    }
                )

        all_results.extend(batch_results)
        batch_count += 1

        save_bad_alignment_visualizations_for_batch(
            batch,
            batch_results,
            batch_idx,
            cfg,
            pixel_threshold=pixel_threshold,
            min_accuracy=min_accuracy,
            output_dir=bad_alignments_dir,
        )

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv_abs, index=False)

    print("\n=== Evaluation Summary ===")
    print(f"Processed {len(all_results)} samples across {batch_count} batches")
    print(f"Results saved to: {output_csv_abs}")

    # Modality-pair breakdown for bad alignments
    if len(bad_samples_records) > 0:
        bad_df = pd.DataFrame(bad_samples_records)
        total_bad = len(bad_df)
        group = (
            bad_df.groupby("modality_pair")
            .size()
            .reset_index(name="bad_count")
            .sort_values("bad_count", ascending=False)
        )
        group["bad_pct"] = group["bad_count"] / float(total_bad) * 100.0
        group["pixel_threshold"] = pixel_threshold

        breakdown_path = output_csv_abs.replace(".csv", f"_modality_breakdown_th{int(pixel_threshold)}.csv")
        group.to_csv(breakdown_path, index=False)
        print(f"Bad-alignment modality breakdown saved to: {breakdown_path}")
    else:
        print("No bad alignments found for the specified threshold/accuracy.")

    # TTA correctness statistics
    if "tta_correct" in df.columns:
        valid_tta = df.dropna(subset=["tta_correct"])
        total_tta = len(valid_tta)
        if total_tta > 0:
            correct_tta = int(valid_tta["tta_correct"].sum())
            tta_accuracy = correct_tta / float(total_tta) * 100.0

            tta_stats = pd.DataFrame(
                {
                    "total_tta_samples": [total_tta],
                    "correct_tta_samples": [correct_tta],
                    "tta_accuracy_pct": [tta_accuracy],
                }
            )
            tta_stats_path = output_csv_abs.replace(".csv", "_tta_stats.csv")
            tta_stats.to_csv(tta_stats_path, index=False)

            print("\n=== TTA Statistics ===")
            print(f"Total TTA-evaluated samples: {total_tta}")
            print(f"Correctly selected augmentations: {correct_tta} ({tta_accuracy:.2f}%)")
            print(f"TTA statistics saved to: {tta_stats_path}")
        else:
            print("No TTA-evaluated samples found.")
    else:
        print("No TTA information recorded in results (tta_correct column missing).")


if __name__ == "__main__":
    main()






