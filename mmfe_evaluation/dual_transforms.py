"""
Paired transforms that apply the same operation to two images simultaneously.
Minimal set needed for CrossOver evaluation (no augmentations).
"""

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class PairToPIL:
    """Convert numpy arrays to PIL Images."""

    def __call__(self, img0, img1=None):
        img0 = Image.fromarray(img0) if not isinstance(img0, Image.Image) else img0
        if img1 is not None:
            img1 = Image.fromarray(img1) if not isinstance(img1, Image.Image) else img1
            return img0, img1
        return img0


class PairResize:
    """Resize both images to the same size."""

    def __init__(self, size, interpolation=TF.InterpolationMode.BICUBIC):
        self.single = T.Resize(size, interpolation=interpolation, antialias=True)

    def __call__(self, img0, img1=None):
        if img1 is not None:
            return self.single(img0), self.single(img1)
        return self.single(img0)


class PairToTensor:
    """Convert both PIL images to tensors."""

    def __init__(self):
        self.single = T.ToTensor()

    def __call__(self, img0, img1=None):
        if img1 is not None:
            return self.single(img0), self.single(img1)
        return self.single(img0)


class PairNormalize:
    """Normalize both tensors with same mean/std."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.single = T.Normalize(mean, std)

    def __call__(self, img0, img1=None):
        if img1 is not None:
            return self.single(img0), self.single(img1)
        return self.single(img0)
