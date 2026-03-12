from hydra.utils import to_absolute_path
import numpy as np
import torch
import os
import matplotlib.cm as mpl_cm
import cv2
import glob
import torch.nn.functional as F

from omegaconf import DictConfig
from typing import Tuple

from dataloading.unified_dataset import UnifiedDataset
from dataloading.dual_transforms import  PairRandomAffine, PairToTensor, PairResize, PairGrayscale, PairNormalize, PairRandomRotation, PairToPIL

def find_latest_checkpoint(checkpoint_dir: str = "./outputs/contrastive/checkpoints") -> str:
    """Find the latest checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by modification time and return the latest
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Using latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def create_datasets(cfg: DictConfig) -> Tuple[UnifiedDataset, UnifiedDataset]:
    """Create training and validation datasets."""
    
    # Data transforms
    # Norm stats
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    if cfg.data.affine_transform.get("common_degrees") is not None:
        dual_transform_train = [
            PairToPIL(),
            PairResize(tuple(cfg.data.image_size)),
            PairGrayscale(num_output_channels=3),
            PairToTensor(),
            PairRandomAffine(degrees=cfg.data.affine_transform.common_degrees, translate=cfg.data.affine_transform.common_translate, scale=cfg.data.affine_transform.common_scale),
            PairNormalize(mean=mean, std=std)
        ]

        dual_transform_val = [
            PairToPIL(),
            PairResize(tuple(cfg.data.image_size)),
            PairGrayscale(num_output_channels=3),
            PairToTensor(),
            PairRandomAffine(degrees=cfg.data.affine_transform.common_degrees, translate=cfg.data.affine_transform.common_translate, scale=cfg.data.affine_transform.common_scale),
            PairNormalize(mean=mean, std=std)
        ]
    else:
        dual_transform_train = [
            PairToPIL(),
            PairResize(tuple(cfg.data.image_size)),
            PairGrayscale(num_output_channels=3),
            PairToTensor(),
            PairNormalize(mean=mean, std=std)
        ]

        dual_transform_val = [
            PairToPIL(),
            PairResize(tuple(cfg.data.image_size)),
            PairGrayscale(num_output_channels=3),
            PairToTensor(),
            PairNormalize(mean=mean, std=std)
        ]

    # Noise transforms
    # Applied after the dual transforms
    if cfg.data.affine_transform.get("noise_degrees") is not None:
        filler = (1-mean)/std
        noise_transform_train = [
            PairRandomAffine(degrees=cfg.data.affine_transform.noise_degrees, translate=cfg.data.affine_transform.noise_translate, scale=cfg.data.affine_transform.noise_scale, filler=filler),
        ]

        noise_transform_val = noise_transform_train
    else:
        noise_transform_train = None
        noise_transform_val = None
    
    # Training dataset configuration
    # Build configs dynamically based on what's available
    train_configs = []
    val_configs = []
    
    # CubiCasa5k dataset
    if cfg.data.get("cubicasa") is not None and cfg.data.cubicasa.path is not None:
        print(f"Creating CubiCasa5k dataset from {cfg.data.cubicasa.path}")
        cubicasa_root = to_absolute_path(cfg.data.cubicasa.path)

        if cfg.data.cubicasa.get("train") is not None and cfg.data.cubicasa.train is not None:
            cubicasa_train_file = to_absolute_path(cfg.data.cubicasa.train)
        else:
            cubicasa_train_file = None

        if cfg.data.cubicasa.get("val") is not None and cfg.data.cubicasa.val is not None:
            cubicasa_val_file = to_absolute_path(cfg.data.cubicasa.val)
        else:
            cubicasa_val_file = None
        
        if cubicasa_train_file is not None:
            train_configs.append({
                "type": "cubicasa5k",
                "args": {
                    "root_dir": cubicasa_root,
                    "sample_ids_file": cubicasa_train_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
        if cubicasa_val_file is not None:
            val_configs.append({
                "type": "cubicasa5k",
                "args": {
                    "root_dir": cubicasa_root,
                    "sample_ids_file": cubicasa_val_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
    
    # Structured3D dataset
    if cfg.data.get("structured3d") is not None and cfg.data.structured3d.path is not None:
        print(f"Creating Structured3D dataset from {cfg.data.structured3d.path}")
        s3d_root = to_absolute_path(cfg.data.structured3d.path)

        if cfg.data.structured3d.get("train") is not None and cfg.data.structured3d.train is not None:
            s3d_train_file = to_absolute_path(cfg.data.structured3d.train)
        else:
            s3d_train_file = None

        if cfg.data.structured3d.get("val") is not None and cfg.data.structured3d.val is not None:
            s3d_val_file = to_absolute_path(cfg.data.structured3d.val)
        else:
            s3d_val_file = None
        
        if s3d_train_file is not None:
            train_configs.append({
                "type": "structured3d",
                "args": {
                    "root_dir": s3d_root,
                    "scene_ids_file": s3d_train_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
        if s3d_val_file is not None:
            val_configs.append({
                "type": "structured3d",
                "args": {
                    "root_dir": s3d_root,
                    "scene_ids_file": s3d_val_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
    
    # Aria SyntheticEnv dataset
    if cfg.data.get("aria_synthenv") is not None and cfg.data.aria_synthenv.path is not None:
        print(f"Creating Aria SyntheticEnv dataset from {cfg.data.aria_synthenv.path}")
        aria_synthenv_root = to_absolute_path(cfg.data.aria_synthenv.path)

        if cfg.data.aria_synthenv.get("train") is not None and cfg.data.aria_synthenv.train is not None:
            aria_synthenv_train_file = to_absolute_path(cfg.data.aria_synthenv.train)
        else:
            aria_synthenv_train_file = None

        if cfg.data.aria_synthenv.get("val") is not None and cfg.data.aria_synthenv.val is not None:
            aria_synthenv_val_file = to_absolute_path(cfg.data.aria_synthenv.val)
        else:
            aria_synthenv_val_file = None
        
        if aria_synthenv_train_file is not None:
            train_configs.append({
                "type": "aria_synthenv",
                "args": {
                    "root_dir": aria_synthenv_root,
                    "scene_ids_file": aria_synthenv_train_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
        if aria_synthenv_val_file is not None:
            val_configs.append({
                "type": "aria_synthenv",
                "args": {
                    "root_dir": aria_synthenv_root,
                    "scene_ids_file": aria_synthenv_val_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
    
    # SwissDwellings dataset
    if cfg.data.get("swiss_dwellings") is not None and cfg.data.swiss_dwellings.path is not None:
        print(f"Creating SwissDwellings dataset from {cfg.data.swiss_dwellings.path}")
        swiss_root = to_absolute_path(cfg.data.swiss_dwellings.path)

        if cfg.data.swiss_dwellings.get("train") is not None and cfg.data.swiss_dwellings.train is not None:
            swiss_train_file = to_absolute_path(cfg.data.swiss_dwellings.train)
        else:
            swiss_train_file = None

        if cfg.data.swiss_dwellings.get("val") is not None and cfg.data.swiss_dwellings.val is not None:
            swiss_val_file = to_absolute_path(cfg.data.swiss_dwellings.val)
        else:
            swiss_val_file = None
        
        if swiss_train_file is not None:
            train_configs.append({
                "type": "swiss_dwellings",
                "args": {
                    "root_dir": swiss_root,
                    "sample_ids_file": swiss_train_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
        if swiss_val_file is not None:
            val_configs.append({
                "type": "swiss_dwellings",
                "args": {
                    "root_dir": swiss_root,
                    "sample_ids_file": swiss_val_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
    
    # Zillow dataset
    if cfg.data.get("zillow") is not None and cfg.data.zillow.path is not None:
        print(f"Creating Zillow dataset from {cfg.data.zillow.path}")
        zillow_root = to_absolute_path(cfg.data.zillow.path)

        if cfg.data.zillow.get("train") is not None and cfg.data.zillow.train is not None:
            zillow_train_file = to_absolute_path(cfg.data.zillow.train)
        else:
            zillow_train_file = None

        if cfg.data.zillow.get("val") is not None and cfg.data.zillow.val is not None:
            zillow_val_file = to_absolute_path(cfg.data.zillow.val)
        else:
            zillow_val_file = None
        
        if zillow_train_file is not None:
            train_configs.append({
                "type": "zillow",
                "args": {
                    "root_dir": zillow_root,
                    "sample_ids_file": zillow_train_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
        if zillow_val_file is not None:
            val_configs.append({
                "type": "zillow",
                "args": {
                    "root_dir": zillow_root,
                    "sample_ids_file": zillow_val_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
    
    # ScanNet dataset
    if cfg.data.get("scannet") is not None and cfg.data.scannet.path is not None:
        print(f"Creating ScanNet dataset from {cfg.data.scannet.path}")
        scannet_root = to_absolute_path(cfg.data.scannet.path)

        if cfg.data.scannet.get("train") is not None and cfg.data.scannet.train is not None:
            scannet_train_file = to_absolute_path(cfg.data.scannet.train)
        else:
            scannet_train_file = None

        if cfg.data.scannet.get("val") is not None and cfg.data.scannet.val is not None:
            scannet_val_file = to_absolute_path(cfg.data.scannet.val)
        else:
            scannet_val_file = None
        
        if scannet_train_file is not None:
            train_configs.append({
                "type": "scannet",
                "args": {
                    "root_dir": scannet_root,
                    "scene_ids_file": scannet_train_file,
                    "image_size": tuple(cfg.data.image_size),
                },
            })
        if scannet_val_file is not None:
            m_pairs = cfg.data.scannet.modality_pairs if cfg.data.scannet.get("modality_pairs", None) is not None else None
            val_configs.append({
                "type": "scannet",
                "args": {
                    "root_dir": scannet_root,
                    "scene_ids_file": scannet_val_file,
                    "image_size": tuple(cfg.data.image_size),
                    "modality_pairs": m_pairs,
                },
            })
    
    # Create datasets
    res = []
    if len(train_configs) > 0:
        res.append(UnifiedDataset(dataset_configs=train_configs, common_transform=dual_transform_train, invertible_transform=noise_transform_train))
    if len(val_configs) > 0:
        res.append(UnifiedDataset(dataset_configs=val_configs, common_transform=dual_transform_val, invertible_transform=noise_transform_val))

    if len(res) == 0:
        raise ValueError("No datasets created")
    elif len(res) == 1:
        return res[0]
    else:
        return res  


def create_val_dataset(cfg: DictConfig) -> UnifiedDataset:
    # Transforms similar to train_contrastive.py (validation path)
    mean = torch.tensor(list(cfg.transforms.mean))
    std = torch.tensor(list(cfg.transforms.std))

    if cfg.transforms.tf_difficulty == "easy":
        transforms = {
            "degrees": 0,
            "translate": [0.0, 0.0],
            "scale": [1.0, 1.0],
        }
    elif cfg.transforms.tf_difficulty == "medium":
        transforms = {
            "degrees": 15,
            "translate": [0.1, 0.1],
            "scale": [0.8, 1.2],
        }
    elif cfg.transforms.tf_difficulty == "hard":
        transforms = {
            "degrees": 180,
            "translate": [0.2, 0.2],
            "scale": [0.6, 1.4],
        }
    elif cfg.transforms.tf_difficulty == "rot_only":
        transforms = {
            "degrees": 180,
            "translate": [0.0, 0.0],
            "scale": [1.0, 1.0],
        }
    else:
        transforms = {
            "degrees": 0,
            "translate": [0.0, 0.0],
            "scale": [1.0, 1.0],
        }

    dual_transform_val = [
        PairToPIL(),
        PairResize(tuple(cfg.data.image_size)),
        PairGrayscale(num_output_channels=3),
        PairToTensor(),
        PairRandomAffine(degrees=180, translate=[0.0, 0.0], scale=[1.0, 1.0]),
        PairNormalize(mean=mean, std=std),
    ]

    filler = (1-mean)/std
    noise_transform_val = [
        PairRandomAffine(degrees=transforms["degrees"], translate=transforms["translate"], scale=transforms["scale"], filler=filler),
    ]

    # Build configs dynamically based on what's available
    val_configs = []
    
    # CubiCasa5k dataset
    if cfg.data.get("cubicasa") is not None and cfg.data.cubicasa.path is not None:
        val_configs.append({
            "type": "cubicasa5k",
            "args": {
                "root_dir": to_absolute_path(cfg.data.cubicasa.path),
                "sample_ids_file": to_absolute_path(cfg.data.cubicasa.val),
                "image_size": tuple(cfg.data.image_size),
            },
        })
    
    # Structured3D dataset
    if cfg.data.get("structured3d") is not None and cfg.data.structured3d.path is not None:
        val_configs.append({
            "type": "structured3d",
            "args": {
                "root_dir": to_absolute_path(cfg.data.structured3d.path),
                "scene_ids_file": to_absolute_path(cfg.data.structured3d.val),
                "image_size": tuple(cfg.data.image_size),
            },
        })
    
    # Aria SyntheticEnv dataset
    if cfg.data.get("aria_synthenv") is not None and cfg.data.aria_synthenv.path is not None:
        val_configs.append({
            "type": "aria_synthenv",
            "args": {
                "root_dir": to_absolute_path(cfg.data.aria_synthenv.path),
                "scene_ids_file": to_absolute_path(cfg.data.aria_synthenv.val),
                "image_size": tuple(cfg.data.image_size),
            },
        })

    # SwissDwellings dataset
    if cfg.data.get("swiss_dwellings") is not None and cfg.data.swiss_dwellings.path is not None:
        val_configs.append({
            "type": "swiss_dwellings",
            "args": {
                "root_dir": to_absolute_path(cfg.data.swiss_dwellings.path),
                "sample_ids_file": to_absolute_path(cfg.data.swiss_dwellings.val),
                "image_size": tuple(cfg.data.image_size),
            },
        })

    # Zillow dataset
    if cfg.data.get("zillow") is not None and cfg.data.zillow.path is not None:
        val_configs.append({
            "type": "zillow",
            "args": {
                "root_dir": to_absolute_path(cfg.data.zillow.path),
                "sample_ids_file": to_absolute_path(cfg.data.zillow.val),
                "image_size": tuple(cfg.data.image_size),
            },
        })

    # ScanNet dataset
    if cfg.data.get("scannet") is not None and cfg.data.scannet.path is not None:
        val_configs.append({
            "type": "scannet",
            "args": {
                "root_dir": to_absolute_path(cfg.data.scannet.path),
                "scene_ids_file": to_absolute_path(cfg.data.scannet.val),
                "image_size": tuple(cfg.data.image_size),
            },
        })

    dataset = UnifiedDataset(dataset_configs=val_configs, common_transform=dual_transform_val, invertible_transform=noise_transform_val)
    return dataset