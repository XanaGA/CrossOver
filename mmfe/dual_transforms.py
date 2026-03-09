import cv2
from omegaconf import ListConfig
import torchvision.transforms as T
from PIL import Image
import torch
import numpy as np
import math
from typing import Tuple, Union, Optional, List
import torch.nn.functional as F
import random

from mmfe_utils.tensor_utils import torch_erode
from .inversible_tf import make_valid_mask

# Required combined transform for random rotation
class PairRandomRotation:
    def __init__(self, degrees, **kwargs):
        self.single = T.RandomRotation(degrees, **kwargs)

    def __call__(self, img0, img1 = None):
        angle = self.single.get_params(self.single.degrees)
        if img1 is not None:
            return T.functional.rotate(img0, angle, fill=1), T.functional.rotate(img1, angle, fill=1)
        else:
            return T.functional.rotate(img0, angle, fill=1)

class PairRandomAffine:
    """
    Apply the same random affine (angle, translate, scale, optional shear) to a pair of images.
    Returns transformed image(s) and (optionally) the sampled transform params so you can undo it
    in feature space.

    Usage:
        transform = PairRandomAffine(degrees=10, translate=(0.02,0.02), scale=(0.95,1.05))
        img0_t, img1_t, params = transform(img0, img1, return_transform=True)
    """
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]] = 0.0,
        translate: Optional[Tuple[float, float]] = None,
        scale: Optional[Tuple[float, float]] = None,
        shear: Optional[Union[float, Tuple[float, float]]] = None,
        interpolation=T.functional.InterpolationMode.BILINEAR,
        filler: Optional[torch.Tensor] = None,
        only_one: bool = False,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.only_one = only_one
        if filler is None:
            filler = torch.tensor([1.0, 1.0, 1.0])
        self.filler = filler


    def _img_size(self, img):
        # img can be PIL Image or tensor C,H,W
        if isinstance(img, Image.Image):
            W, H = img.size
        elif torch.is_tensor(img):
            if img.dim() == 3:
                _, H, W = img.shape
            elif img.dim() == 4:  # unlikely here, but just in case (N,C,H,W)
                _, _, H, W = img.shape
            else:
                raise ValueError("Unsupported tensor image shape")
        else:
            raise TypeError("Image must be PIL Image or torch.Tensor")
        return int(W), int(H)

    def _sample(self, img):
        W, H = self._img_size(img)

        # angle
        if isinstance(self.degrees, (tuple, list, ListConfig)):
            angle = random.uniform(self.degrees[0], self.degrees[1])
        else:
            angle = random.uniform(-float(self.degrees), float(self.degrees))

        # translate in pixels
        if self.translate is None:
            tx, ty = 0.0, 0.0
        else:
            max_dx = float(self.translate[0]) * W
            max_dy = float(self.translate[1]) * H
            tx = random.uniform(-max_dx, max_dx)
            ty = random.uniform(-max_dy, max_dy)

        # scale
        if self.scale is None:
            scale = 1.0
        else:
            if isinstance(self.scale, (tuple, list, ListConfig)):
                scale = random.uniform(self.scale[0], self.scale[1])
            else:
                scale = float(self.scale)

        # shear (single angle or tuple)
        if self.shear is None:
            shear = 0.0
        else:
            if isinstance(self.shear, (tuple, list, ListConfig)):
                shear = random.uniform(self.shear[0], self.shear[1])
            else:
                shear = float(self.shear)

        return dict(angle=float(angle),
                    translate=(float(tx), float(ty)),
                    scale=float(scale),
                    shear=float(shear),
                    image_size=(W, H))

    def __call__(self, img0, img1=None, return_transform: bool = False, batch_size: int = 1):
        """
        img0: PIL Image or torch.Tensor (C,H,W)
        img1: optionally second image
        if return_transform: returns (img0_t, img1_t, params) else returns transformed images only
        """
        device = img0.device
        params = self._sample(img0)
        angle = params["angle"]
        translate = params["translate"]
        scale = params["scale"]
        shear = params["shear"]
        

        # Use torchvision functional affine which accepts tensors or PIL
        valid_mask = make_valid_mask(params, device=img0.device, dtype=img0.dtype)
        valid_mask = torch_erode(valid_mask, kernel_size=3, iterations=1)
        filler = self.filler.view(-1, 1, 1)

        img0_t = T.functional.affine(img0, angle=angle, translate=translate, scale=scale,
                           shear=shear, interpolation=self.interpolation)

        img0_t = torch.where(~valid_mask.bool(), filler.to(device), img0_t)
        
        if img1 is not None and not self.only_one:
            img1_t = T.functional.affine(img1, angle=angle, translate=translate, scale=scale,
                               shear=shear, interpolation=self.interpolation)
            
            img1_t = torch.where(~valid_mask.bool(), filler.to(device), img1_t)

        elif img1 is not None and self.only_one:
            img1_t = img1
        else:
            img1_t = None


        if batch_size > 1:
            valid_mask = valid_mask.repeat(batch_size, 1, 1, 1)
            params["angle"] = torch.tensor([params["angle"]], device=device).repeat(batch_size, 1) 
            trans_tensor = torch.tensor([params["translate"]], device=device).repeat(batch_size, 1)          
            params["translate"] = [trans_tensor[:, 0], trans_tensor[:, 1]]
            params["scale"] = torch.tensor([params["scale"]], device=device).repeat(batch_size, 1)
            params["shear"] = torch.tensor([params["shear"]], device=device).repeat(batch_size, 1) 
            fill = torch.tensor([fill], device=device).repeat(batch_size, 1)
        params["valid_mask"] = valid_mask

        if return_transform:
            if img1 is not None:
                return img0_t, img1_t, params
            else:
                return img0_t, params
        else:
            if img1 is not None:
                return img0_t, img1_t
            else:
                return img0_t

# Naive implementation that applies the same transform to both images
class PairResize:
    def __init__(self, size, **kwargs):
        self.single = T.Resize(size, **kwargs)

    def __call__(self, img0, img1 = None):
        if img1 is not None:
            return self.single(img0), self.single(img1)
        else:
            return self.single(img0)

class PairToTensor:
    def __init__(self, **kwargs):
        self.single = T.ToTensor(**kwargs)

    def __call__(self, img0, img1 = None):
        if img1 is not None:
            return self.single(img0), self.single(img1)
        else:
            return self.single(img0)

class PairNormalize:
    def __init__(self, mean, std, **kwargs):
        self.mean = mean
        self.std = std
        self.single = T.Normalize(mean, std, **kwargs)

    def __call__(self, img0, img1 = None):
        if img1 is not None:
            return self.single(img0), self.single(img1)
        else:
            return self.single(img0)

class PairGrayscale:
    def __init__(self, num_output_channels=3, **kwargs):
        self.single = T.Grayscale(num_output_channels, **kwargs)

    def __call__(self, img0, img1 = None):
        if img1 is not None:
            return self.single(img0), self.single(img1)
        else:
            return self.single(img0)

class PairToPIL:
    def __init__(self):
        pass

    def __call__(self, img0, img1 = None):
        if img1 is not None:
            return Image.fromarray(img0), Image.fromarray(img1)
        else:
            return Image.fromarray(img0)
