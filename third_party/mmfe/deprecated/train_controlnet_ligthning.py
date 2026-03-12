#!/usr/bin/env python3
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import time
from typing import Tuple, List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from dataloading.unified_dataset import UnifiedDataset
from dataloading.dual_transforms import (
    PairToPIL,
    PairResize,
    PairGrayscale,
    PairToTensor,
)

from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
)

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path


def build_unified_datasets(
    cubicasa_path: str,
    cubicasa_ids: str,
    s3d_path: str,
    s3d_ids: str,
    image_size: Tuple[int, int],
) -> Tuple[UnifiedDataset, UnifiedDataset]:
    transform = [
        PairToPIL(),
        PairResize(tuple(image_size)),
        PairGrayscale(num_output_channels=3),
        PairToTensor(),
    ]

    train_cfgs: List[Dict[str, Any]] = []
    if cubicasa_path:
        train_cfgs.append({
            "type": "cubicasa5k",
            "args": {
                "root_dir": cubicasa_path,
                "sample_ids_file": cubicasa_ids,
                "image_size": tuple(image_size),
                "dual_transform": transform,
            },
        })
    if s3d_path:
        train_cfgs.append({
            "type": "structured3d",
            "args": {
                "root_dir": s3d_path,
                "scene_ids_file": s3d_ids,
                "image_size": tuple(image_size),
                "dual_transform": transform,
            },
        })

    if len(train_cfgs) == 0:
        raise ValueError("Provide at least one dataset path via --cubicasa-path or --structured3d-path")

    # Use same config for val for now
    val_cfgs = train_cfgs

    train_ds = UnifiedDataset(dataset_configs=train_cfgs, common_transform=None, invertible_transform=None, text_description=True)
    val_ds = UnifiedDataset(dataset_configs=val_cfgs, common_transform=None, invertible_transform=None, text_description=True)
    return train_ds, val_ds


def import_text_encoder_cls(model_name: str, revision: str = None):
    cfg = PretrainedConfig.from_pretrained(model_name, subfolder="text_encoder", revision=revision)
    if cfg.architectures[0] == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    raise ValueError(f"Unsupported text encoder: {cfg.architectures[0]}")


@hydra.main(config_path="../../configs", config_name="train_controlnet_ligthning", version_base="1.3")
def main(cfg: DictConfig):
    # Map contrastive-like config to controlnet fields
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defaults
    pretrained_model = cfg.model.get("pretrained", "runwayml/stable-diffusion-v1-5") if "model" in cfg else "runwayml/stable-diffusion-v1-5"
    controlnet_model = cfg.model.get("controlnet", "") if "model" in cfg else ""
    revision = cfg.model.get("revision", None) if "model" in cfg else None
    variant = cfg.model.get("variant", None) if "model" in cfg else None

    image_size = tuple(cfg.data.image_size) if "data" in cfg and "image_size" in cfg.data else (512, 512)

    cubicasa_path = to_absolute_path(cfg.data.cubicasa.path) if "data" in cfg and "cubicasa" in cfg.data else ""
    cubicasa_ids = to_absolute_path(cfg.data.cubicasa.train) if "data" in cfg and "cubicasa" in cfg.data else ""
    s3d_path = to_absolute_path(cfg.data.structured3d.path) if "data" in cfg and "structured3d" in cfg.data else ""
    s3d_ids = to_absolute_path(cfg.data.structured3d.train) if "data" in cfg and "structured3d" in cfg.data else ""

    batch_size = int(cfg.train.batch_size) if "train" in cfg and "batch_size" in cfg.train else 2
    num_workers = int(cfg.train.num_workers) if "train" in cfg and "num_workers" in cfg.train else 4
    epochs = int(cfg.train.epochs) if "train" in cfg and "epochs" in cfg.train else 1
    lr = float(cfg.optim.lr) if "optim" in cfg and "lr" in cfg.optim else 5e-6
    max_steps = int(cfg.train.max_steps) if "train" in cfg and "max_steps" in cfg.train else 0
    grad_accum = int(cfg.train.accumulate_grad_batches) if "train" in cfg and "accumulate_grad_batches" in cfg.train else 1
    max_grad_norm = float(cfg.train.max_grad_norm) if "train" in cfg and "max_grad_norm" in cfg.train else 1.0
    seed = int(cfg.train.seed) if "train" in cfg and "seed" in cfg.train else 123
    mixed_precision = cfg.train.get("mixed_precision", "no") if "train" in cfg else "no"
    enable_xformers = bool(cfg.train.get("enable_xformers", True)) if "train" in cfg else True
    gradient_checkpointing = bool(cfg.train.get("gradient_checkpointing", True)) if "train" in cfg else True
    allow_tf32 = bool(cfg.train.get("allow_tf32", True)) if "train" in cfg else True
    use_8bit_adam = bool(cfg.train.get("use_8bit_adam", False)) if "train" in cfg else False

    output_dir = to_absolute_path(cfg.logging.output_dir) if "logging" in cfg and "output_dir" in cfg.logging else \
        "/local/home/xanadon/mmfe/outputs/controlnet"
    use_wandb = bool(cfg.logging.wandb) if "logging" in cfg and "wandb" in cfg.logging else False
    wandb_project = cfg.logging.get("project_name", "controlnet-unified") if "logging" in cfg else "controlnet-unified"
    wandb_run = cfg.logging.get("experiment_name", "train") if "logging" in cfg else "train"
    log_interval = int(cfg.logging.get("log_interval", 50)) if "logging" in cfg else 50
    val_interval = int(cfg.logging.get("val_interval", 500)) if "logging" in cfg else 500

    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Data
    train_ds, val_ds = build_unified_datasets(
        cubicasa_path, cubicasa_ids, s3d_path, s3d_ids, image_size
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    # Tokenizer / text encoder
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer", revision=revision, use_fast=False)
    TextEncoder = import_text_encoder_cls(pretrained_model, revision)
    text_encoder = TextEncoder.from_pretrained(pretrained_model, subfolder="text_encoder", revision=revision, variant=variant)

    # Diffusion models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision, variant=variant)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet", revision=revision, variant=variant)
    if controlnet_model:
        controlnet = ControlNetModel.from_pretrained(controlnet_model)
    else:
        controlnet = ControlNetModel.from_unet(unet)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # Precision
    dtype = torch.float32
    if mixed_precision == "fp16":
        dtype = torch.float16
    elif mixed_precision == "bf16":
        dtype = torch.bfloat16

    vae.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    controlnet.to(device, dtype=dtype)

    # Memory optimizations
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
    if enable_xformers:
        try:
            import xformers  # noqa: F401
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # Optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb  # type: ignore
            optimizer = bnb.optim.AdamW8bit(controlnet.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
        except Exception:
            optimizer = torch.optim.AdamW(controlnet.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    # W&B
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run, config=OmegaConf.to_container(cfg, resolve=True))

    global_step = 0
    scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision == "fp16"))

    def encode_text(prompts: List[str]):
        input_ids = tokenizer(
            prompts,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad():
            enc = text_encoder(input_ids, return_dict=False)[0]
        return enc

    for epoch in range(epochs):
        for batch in train_loader:
            control_images = batch["modality_0"].to(device, dtype=dtype).contiguous(memory_format=torch.channels_last)
            target_images = batch["modality_1"].to(device, dtype=dtype).contiguous(memory_format=torch.channels_last)
            prompts = batch.get("m1_description", None)
            if prompts is None:
                prompts = [batch.get("modality_desc", "An image of an apartment.")] * control_images.shape[0]

            # VAE encode targets to latents
            with torch.no_grad():
                latents = vae.encode(target_images).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(dtype=dtype)

            encoder_hidden_states = encode_text(prompts)
            cond_image = control_images

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(mixed_precision in ["fp16", "bf16"])):
                down_res, mid_res = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=cond_image,
                    return_dict=False,
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[s.to(dtype=dtype) for s in down_res],
                    mid_block_additional_residual=mid_res.to(dtype=dtype),
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Gradient accumulation to reduce peak memory
            loss_to_backprop = loss / max(1, grad_accum)
            scaler.scale(loss_to_backprop).backward()
            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if use_wandb and (global_step % log_interval == 0):
                wandb.log({"train/loss": loss.item(), "train/step": global_step})

            if max_steps and global_step >= max_steps:
                break

        # Save at each epoch
        save_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        controlnet.save_pretrained(save_dir)

        if use_wandb:
            wandb.log({"epoch": epoch + 1})

        if max_steps and global_step >= max_steps:
            break

    # Final save
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    controlnet.save_pretrained(final_dir)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()