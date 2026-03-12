"""
Configuration module for contrastive learning training.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    model_type: str = "dual_modality"
    backbone_name: str = "resnet18"
    projection_dim: int = 128
    pretrained: bool = True
    freeze_backbone: bool = False
    projection_head_type: str = "mlp"
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """Configuration for the loss function."""
    loss_type: str = "infonce"
    temperature: float = 0.07
    margin: float = 1.0
    reduction: str = "mean"


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    scheduler: bool = False
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    batch_size: int = 32
    num_workers: int = 4
    image_size: tuple = (256, 256)
    pin_memory: bool = True
    drop_last: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    gpus: int = 1
    precision: int = 16
    accumulate_grad_batches: int = 1
    deterministic: bool = False
    seed: int = 42
    val_check_interval: float = 0.5
    log_every_n_steps: int = 10


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing."""
    output_dir: str = "./outputs"
    experiment_name: str = "contrastive_learning"
    wandb: bool = False
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and splits."""
    cubicasa_path: str = ""
    structured3d_path: str = ""
    cubicasa_train: str = ""
    cubicasa_val: str = ""
    structured3d_train: str = ""
    structured3d_val: str = ""


@dataclass
class FullConfig:
    """Complete configuration for training."""
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "loss": self.loss.__dict__,
            "optimizer": self.optimizer.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
            "dataset": self.dataset.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FullConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                setattr(config.model, key, value)
        
        if "loss" in config_dict:
            for key, value in config_dict["loss"].items():
                setattr(config.loss, key, value)
        
        if "optimizer" in config_dict:
            for key, value in config_dict["optimizer"].items():
                setattr(config.optimizer, key, value)
        
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                setattr(config.data, key, value)
        
        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                setattr(config.training, key, value)
        
        if "logging" in config_dict:
            for key, value in config_dict["logging"].items():
                setattr(config.logging, key, value)
        
        if "dataset" in config_dict:
            for key, value in config_dict["dataset"].items():
                setattr(config.dataset, key, value)
        
        return config


def get_default_config() -> FullConfig:
    """Get default configuration."""
    return FullConfig()


def get_resnet50_config() -> FullConfig:
    """Get configuration optimized for ResNet50."""
    config = get_default_config()
    config.model.backbone_name = "resnet50"
    config.model.projection_dim = 256
    config.optimizer.lr = 5e-5
    config.optimizer.weight_decay = 1e-4
    config.data.batch_size = 16
    return config


def get_efficientnet_config() -> FullConfig:
    """Get configuration optimized for EfficientNet."""
    config = get_default_config()
    config.model.backbone_name = "efficientnet_b0"
    config.model.projection_dim = 128
    config.optimizer.lr = 1e-4
    config.optimizer.weight_decay = 1e-5
    config.data.batch_size = 32
    return config


def get_fast_training_config() -> FullConfig:
    """Get configuration for fast training/testing."""
    config = get_default_config()
    config.model.backbone_name = "resnet18"
    config.training.epochs = 10
    config.data.batch_size = 64
    config.training.val_check_interval = 1.0
    return config
