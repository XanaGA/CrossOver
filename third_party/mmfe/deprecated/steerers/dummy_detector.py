from typing import Union
import torch

from steerers.steerers_utils import rotate_keypoints, filter_kpts_inside_image, visualize_keypoint_rotation


class DummyDetector:
    """
    A simple detector that samples random 2D keypoints uniformly in normalized
    grid coordinates [-1, 1].

    Expected batch keys (at least one pair present):
      - ("im_A", "im_B") with shape [B, C, H, W]
      - ("modality_0", "modality_1") with shape [B, C, H, W]

    Returns a dict with key "keypoints" of shape [2*B, N, 2] in normalized coords.
    The first B entries correspond to A, the next B to B.
    """

    def __init__(self, seed: int | None = None):
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    @torch.no_grad()
    def detect(self, batch: dict, num_keypoints: int = 1000, rot_A_to_B: Union[int, float, None] = None, debug: bool = False) -> dict:
        if "im_A" in batch and "im_B" in batch:
            im_A = batch["im_A"]
            im_B = batch["im_B"]
        elif "modality_0" in batch and "modality_1" in batch:
            im_A = batch["modality_0"]
            im_B = batch["modality_1"]
        else:
            raise KeyError("DummyDetector expects ('im_A','im_B') or ('modality_0','modality_1') in batch")

        if im_A.dim() != 4 or im_B.dim() != 4:
            raise ValueError("Images must be [B, C, H, W]")

        if im_A.shape[0] != im_B.shape[0]:
            raise ValueError("A and B must have the same batch size")

        batch_size = im_A.shape[0]
        device = im_A.device

        # Sample uniform points in normalized coordinate space [-1, 1]
        # Shape: [B, N, 2]
        kpts_A = torch.rand((batch_size, num_keypoints, 2), device=device, generator=self.generator) * 2.0 - 1.0

        if rot_A_to_B is not None:
            kpts_B = rotate_keypoints(kpts_A, rot_A_to_B, continuous_rot=False, debug=debug)
        else:
            kpts_B = kpts_A 

        # Filter by image bounds of B
        kpts_A, kpts_B = filter_kpts_inside_image(kpts_A, kpts_B, im_B, min_keep=1, random_select=True)

        # Concatenate along the leading dimension to match downstream expectations
        # Final shape: [2*B, N_filtered, 2]
        keypoints = torch.cat([kpts_A, kpts_B], dim=0)

        return {"keypoints": keypoints}

    @torch.no_grad()
    def detect_on_images(self, im_A: torch.Tensor, im_B: torch.Tensor, num_keypoints: int = 1000) -> dict:
        batch = {"im_A": im_A, "im_B": im_B}
        return self.detect(batch, num_keypoints=num_keypoints)


