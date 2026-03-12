import torch
import torchvision.transforms.functional as TTF
from tqdm import tqdm
import random
import numpy as np
import cv2
import wandb

from dataloading.inversible_tf import make_valid_mask
from steerers.steerers_utils import dict_to_device
from mmfe_utils.tensor_utils import torch_erode


def train_step(train_batch,
               model,
               objective,
               optimizer,
               grad_scaler=None,
               generator_rot= None,
               rot_cont=False,
               steerer=None,
               debug=False,
               **kwargs):
    optimizer.zero_grad()
    if generator_rot is not None:
        nbr_rot_A = random.randint(0, (360 // generator_rot)-1)
        nbr_rot_B = random.randint(0, (360 // generator_rot)-1)
        while nbr_rot_A == nbr_rot_B:
            nbr_rot_B = random.randint(0, (360 // generator_rot)-1)
        rot_deg_A = (nbr_rot_A * generator_rot) % 360
        rot_deg_B = (nbr_rot_B * generator_rot) % 360
        train_batch["im_A"] = TTF.rotate(
            train_batch["modality_0"],
            rot_deg_A,
            interpolation=TTF.InterpolationMode.BILINEAR,
        )
        train_batch["im_B"] = TTF.rotate(
            train_batch["modality_1"],
            rot_deg_B,
            interpolation=TTF.InterpolationMode.BILINEAR,
        )
        valid_mask_A = make_valid_mask({"angle": rot_deg_A, "translate": [0, 0], "scale": 1, "shear": 0, "image_size": train_batch["im_A"].shape[-2:]}, 
                            device=train_batch["im_A"].device, dtype=train_batch["im_A"].dtype)
        valid_mask_B = make_valid_mask({"angle": rot_deg_B, "translate": [0, 0], "scale": 1, "shear": 0, "image_size": train_batch["im_B"].shape[-2:]}, 
                            device=train_batch["im_B"].device, dtype=train_batch["im_B"].dtype)

        valid_mask_A = torch_erode(valid_mask_A, kernel_size=3, iterations=1)
        valid_mask_B = torch_erode(valid_mask_B, kernel_size=3, iterations=1)

        train_batch["im_A"] = torch.where(~valid_mask_A.bool(), 1, train_batch["im_A"])
        train_batch["im_B"] = torch.where(~valid_mask_B.bool(), 1, train_batch["im_B"])

        train_batch["rot_deg_A"] = rot_deg_A
        train_batch["rot_deg_B"] = rot_deg_B
        train_batch["rot_deg_A_to_B"] = rot_deg_B -rot_deg_A
        train_batch["generator_rot"] = generator_rot

        if debug:
            # Show original and rotated images A and B side by side using opencv2
            def denorm(img: torch.Tensor) -> np.ndarray:
                # img: [C,H,W], ImageNet stats
                mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(3, 1, 1)
                std = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(3, 1, 1)
                x = (img * std + mean).clamp(0, 1)
                x = (x.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                return x

            try:
                img0 = train_batch["modality_0"][0]
                img1 = train_batch["modality_1"][0]
                imgA = train_batch["im_A"][0]
                imgB = train_batch["im_B"][0]

                vis0 = denorm(img0)
                vis1 = denorm(img1)
                visA = denorm(imgA)
                visB = denorm(imgB)

                top = np.concatenate([vis0, vis1], axis=1)
                bottom = np.concatenate([visA, visB], axis=1)
                grid = np.concatenate([top, bottom], axis=0)

                rot_deg_A_to_B = train_batch["rot_deg_A_to_B"]
                cv2.imshow(f"Rotations: A = {rot_deg_A}, B = {rot_deg_B} | A to B = {rot_deg_A_to_B}", grid)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()
            except Exception:
                pass

        out = model(train_batch)
        l = objective(out,
                      train_batch,
                      nbr_rot_A,
                      nbr_rot_B,
                      steerer=steerer)
    elif rot_cont:
        rot_A = 2 * np.pi * random.random()
        rot_B = 2 * np.pi * random.random()
        train_batch["im_A"] = TTF.rotate(
            train_batch["im_A"],
            np.rad2deg(rot_A),
            interpolation=TTF.InterpolationMode.BILINEAR,
        )
        train_batch["im_B"] = TTF.rotate(
            train_batch["im_B"],
            np.rad2deg(rot_B),
            interpolation=TTF.InterpolationMode.BILINEAR,
        )
        out = model(train_batch)
        l = objective(out,
                      train_batch,
                      rot_A,
                      rot_B,
                      steerer=steerer,
                      continuous_rot=True)
    else:
        out = model(train_batch)
        l = objective(out,
                      train_batch)
    if grad_scaler is not None:
        grad_scaler.scale(l).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        l.backward()
        optimizer.step()
    return {"train_out": out, "train_loss": l.item()}


def train_one_epoch(
                  dataloader,
                  model,
                  objective,
                  optimizer,
                  lr_scheduler,
                  grad_scaler = None,
                  generator_rot=None,
                  steerer=None,
                  rot_cont=False,
                  progress_bar=True,
                  log_wandb=False):
    if rot_cont and generator_rot is not None:
        raise ValueError()
    
    losses = []
    for batch in tqdm(dataloader, disable=not progress_bar, mininterval = 10.):
        batch = dict_to_device(batch)
        model.train(True)
        # batch = to_best_device(batch)
        step_result = train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler = grad_scaler,
            generator_rot = generator_rot,
            steerer = steerer,
            rot_cont = rot_cont,
        )
        losses.append(step_result["train_loss"])
        lr_scheduler.step()

        # Log training loss to wandb
        if log_wandb:
            wandb.log({"train_loss": step_result["train_loss"]})
    
    return losses