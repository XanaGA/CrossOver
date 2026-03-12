import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from steerers.steerers_utils import visualize_image_and_keypoints


def dual_log_softmax_matcher(desc_A: torch.Tensor, desc_B: torch.Tensor, inv_temperature = 1, normalize = False):
    if normalize:
        desc_A = desc_A/desc_A.norm(dim=-1,keepdim=True)
        desc_B = desc_B/desc_B.norm(dim=-1,keepdim=True)
    corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    logP = corr.log_softmax(dim = -2) + corr.log_softmax(dim= -1)
    return logP

class DescriptorLoss(nn.Module):
    
    def __init__(self,
                 detector,
                 num_keypoints = 5000,
                 normalize_descriptions = False,
                 inv_temp = 1,
                 device = None) -> None:
        super().__init__()
        self.detector = detector
        self.tracked_metrics = {}
        self.num_keypoints = num_keypoints
        self.normalize_descriptions = normalize_descriptions
        self.inv_temp = inv_temp
    
    # def warp_from_depth(self, batch, kpts_A, kpts_B):
    #     mask_A_to_B, kpts_A_to_B = warp_kpts(kpts_A, 
    #                 batch["im_A_depth"],
    #                 batch["im_B_depth"],
    #                 batch["T_1to2"],
    #                 batch["K1"],
    #                 batch["K2"],)
    #     mask_B_to_A, kpts_B_to_A = warp_kpts(kpts_B, 
    #                 batch["im_B_depth"],
    #                 batch["im_A_depth"],
    #                 batch["T_1to2"].inverse(),
    #                 batch["K2"],
    #                 batch["K1"],)
    #     return (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A)
    
    # def warp_from_homog(self, batch, kpts_A, kpts_B):
    #     kpts_A_to_B = homog_transform(batch["Homog_A_to_B"], kpts_A)
    #     kpts_B_to_A = homog_transform(batch["Homog_A_to_B"].inverse(), kpts_B)
    #     return (None, kpts_A_to_B), (None, kpts_B_to_A)

    def supervised_loss(self,
                        outputs,
                        batch,
                        rot_A=0,
                        rot_B=0,
                        steerer=None,
                        continuous_rot=False,
                        debug=False,
                       ):

        kpts_A, kpts_B = self.detector.detect(batch, num_keypoints = self.num_keypoints, 
                                                rot_A_to_B=batch.get("rot_deg_A_to_B", None), debug=debug)['keypoints'].clone().chunk(2)

        desc_grid_A, desc_grid_B = outputs["description_grid"].chunk(2)

        desc_A = F.grid_sample(desc_grid_A.float(), kpts_A[:,None], mode = "bilinear", align_corners = False)[:,:,0].mT
        desc_B = F.grid_sample(desc_grid_B.float(), kpts_B[:,None], mode = "bilinear", align_corners = False)[:,:,0].mT


        # Matching indices - i-th keypoint in A corresponds to i-th keypoint in B
        batch_size, num_kpts = kpts_A.shape[:2]
        batch_indices = torch.arange(batch_size, device=kpts_A.device).unsqueeze(1).expand(-1, num_kpts).flatten()
        kpt_indices_A = torch.arange(num_kpts, device=kpts_A.device).unsqueeze(0).expand(batch_size, -1).flatten()
        kpt_indices_B = torch.arange(num_kpts, device=kpts_B.device).unsqueeze(0).expand(batch_size, -1).flatten()
        inds = torch.stack([batch_indices, kpt_indices_A, kpt_indices_B], dim=1)
        

        # cosA, sinA = np.cos(rot_A_rad), np.sin(rot_A_rad)
        # cosB, sinB = np.cos(rot_B_rad), np.sin(rot_B_rad)
        # R_A_transpose = torch.tensor([[cosA, sinA],
        #                                 [-sinA, cosA]],
        #                                 dtype=kpts_A.dtype,
        #                                 device=kpts_A.device)
        # R_B_transpose = torch.tensor([[cosB, sinB],
        #                                 [-sinB, cosB]],
        #                                 dtype=kpts_B.dtype,
        #                                 device=kpts_B.device)
        # kpts_A = kpts_A @ R_A_transpose
        # kpts_B = kpts_B @ R_B_transpose

        # if "im_A_depth" in batch:
        #     (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A) = self.warp_from_depth(batch, kpts_A, kpts_B)
        # elif "Homog_A_to_B" in batch:
        #     (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A) = self.warp_from_homog(batch, kpts_A, kpts_B)
        
        if steerer is not None and (rot_A > 0 or rot_B > 0):
            if continuous_rot:
                rot_A_rad = rot_A
                rot_B_rad = rot_B
                rot = rot_A_rad - rot_B_rad
                if rot < 0:
                    rot = rot + 2 * np.pi
                desc_B = steerer(desc_B, rot)
            else:
                nbr_rot_A = rot_A
                nbr_rot_B = rot_B
                max_turns = 360 // batch["generator_rot"]
                nbr_rot = (max_turns + nbr_rot_A - nbr_rot_B) % max_turns  # nbr of rotations to align B with A
                # nbr_rot = (abs(max_turns + nbr_rot_A - nbr_rot_B)) % max_turns  # nbr of rotations to align B with A
                for _ in range(nbr_rot):
                    desc_B = steerer(desc_B)


        if debug:
            print(f"nbr_rot_A: {nbr_rot_A}, nbr_rot_B: {nbr_rot_B}")
            print(f"max_turns: {max_turns}")
            print(f"nbr_rot: {nbr_rot}")
            
            visualize_image_and_keypoints(batch, kpts_A, kpts_B, nbr_rot_A, nbr_rot_B, nbr_rot)

        logP_A_B = dual_log_softmax_matcher(desc_A, desc_B, 
                                            normalize = self.normalize_descriptions,
                                            inv_temperature = self.inv_temp)
        neg_log_likelihood = -logP_A_B[inds[:,0], inds[:,1], inds[:,2]].mean()

        self.tracked_metrics["neg_log_likelihood"] = (
            0.99 * self.tracked_metrics.get("neg_log_likelihood", neg_log_likelihood.detach().item())
            + 0.01 * neg_log_likelihood.detach().item()
        )
        if np.random.rand() > 0.99:
            print(f'nll: {self.tracked_metrics["neg_log_likelihood"]}')

        loss = neg_log_likelihood

        return loss
    
    def forward(self,
                outputs,
                batch,
                rot_A=0,
                rot_B=0,
                steerer=None,
                continuous_rot=False,
                debug=False,
               ):
        losses = self.supervised_loss(outputs,
                                      batch,
                                      rot_A,
                                      rot_B,
                                      steerer=steerer,
                                      continuous_rot=continuous_rot,
                                      debug=debug,
                                     )
        return losses