import os
import argparse
import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dataloading.unified_dataset import UnifiedDataset
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from dataloading.dual_transforms import PairToPIL, PairResize, PairGrayscale, PairToTensor
from mmfe_utils.controlNetWaffle import preprocess_control_image, to_pil





def build_dataset(args):
    # Match the unified dataloader test transforms
    transform = [
        PairToPIL(),
        PairResize((512, 512)) if args.image_size is None else PairResize(args.image_size),
        PairGrayscale(num_output_channels=3),
        PairToTensor(),
    ]

    dataset_configs = []
    if args.cubicasa_path:
        if not os.path.exists(args.cubicasa_path):
            raise FileNotFoundError(f"CubiCasa5k path does not exist: {args.cubicasa_path}")
        dataset_configs.append({
            "type": "cubicasa5k",
            "args": {
                "root_dir": args.cubicasa_path,
                "sample_ids_file": args.cubicasa_ids,
                "image_size": args.image_size,
                "dual_transform": transform,
                # "modality_pairs": [("gt_svg_annotations", "drawing"), ("gt_svg_annotations", "lidar_points")],
                "modality_pairs": [ ("lidar_points", "gt_svg_annotations")],
            },
        })

    if args.structured3d_path:
        if not os.path.exists(args.structured3d_path):
            raise FileNotFoundError(f"Structured3D path does not exist: {args.structured3d_path}")
        dataset_configs.append({
            "type": "structured3d",
            "args": {
                "root_dir": args.structured3d_path,
                "scene_ids_file": args.structured3d_ids,
                "no_color": args.no_color,
                "image_size": args.image_size,
                "dpi": args.dpi,
                "dual_transform": transform,
            },
        })

    if len(dataset_configs) == 0:
        raise ValueError("Provide at least one dataset path via --cubicasa-path or --structured3d-path")

    ds = UnifiedDataset(dataset_configs=dataset_configs, common_transform=None, invertible_transform=None, text_description=True)
    return ds


def main():
    parser = argparse.ArgumentParser(description="ControlNet test: control=modality_0, prompt=m1_description")
    # Match test_unified_dataloader interface
    parser.add_argument("--cubicasa-path", type=str, help="Path to CubiCasa5k dataset root directory (optional)")
    parser.add_argument("--cubicasa-ids", type=str, help="Optional text file with CubiCasa5k IDs to include")
    parser.add_argument("--structured3d-path", type=str, help="Path to Structured3D dataset root directory (optional)")
    parser.add_argument("--structured3d-ids", type=str, help="Optional json file with Structured3D scene IDs to include")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), help="Resize images to HEIGHT x WIDTH")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for Structured3D rendering (default: 100)")
    parser.add_argument("--no-color", action="store_true", help="Generate grayscale/no-color for Structured3D")
    parser.add_argument("--index", type=int, default=0, help="Sample index in the unified dataset")
    parser.add_argument("--controlnet", type=str, default="lllyasviel/sd-controlnet-canny", help="ControlNet model id")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion base model id")
    parser.add_argument("--steps", type=int, default=30, help="Num inference steps")
    parser.add_argument("--guidance", type=float, default=15.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--outdir", type=str, default="/local/home/xanadon/mmfe/outputs/visualizations", help="Output directory to save the panel")
    parser.add_argument("--height", type=int, default=512, help="Generation height")
    parser.add_argument("--width", type=int, default=512, help="Generation width")
    parser.add_argument("--num-examples", type=int, default=1, help="How many samples to generate from the dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset
    dataset = build_dataset(args)

    # Load ControlNet pipeline once
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.sd_model,
        controlnet=controlnet,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None
    )
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention() if device.type == "cuda" else None
    pipe = pipe.to(device)

    # Iterate and generate
    os.makedirs(args.outdir, exist_ok=True)
    max_n = min(len(dataset), max(0, int(args.num_examples)))
    start_idx = int(args.index)
    end_idx = min(len(dataset), start_idx + max_n)

    for i in range(start_idx, end_idx):
        item = dataset[i]

        if not "floorplan" in args.controlnet:
            control_img_pil = to_pil(item["modality_0"]).convert("RGB").resize((args.width, args.height), Image.BILINEAR)
        else:
            control_img_pil = preprocess_control_image(item["modality_0"], args.width, args.height, red_threshold=0.005)

        gt_img_pil = to_pil(item["modality_1"]).convert("RGB")

        prompt = item.get("m1_description", None)
        if prompt is None:
            prompt = item.get("modality_desc", "An image of a furnished apartment.")

        prompt = "A floorplan of a school."

        generator = torch.Generator(device=device).manual_seed(args.seed)
        out = pipe(
            prompt=prompt,
            image=control_img_pil,
            controlnet_conditioning_scale=1.0,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            height=args.height,
            width=args.width,
        )
        gen_img_pil = out.images[0]

        # Build a panel: control image, prompt text, generated image, ground truth
        source = item.get("source_dataset", "mixed")
        out_path = os.path.join(args.outdir, f"controlnet_demo_{source}_{i}.png")

        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(control_img_pil)
        ax1.set_title("Control (modality_0)")
        ax1.axis("off")

        ax2 = plt.subplot(2, 2, 2)
        ax2.text(0.5, 0.5, prompt, wrap=True, ha='center', va='center', fontsize=12)
        ax2.set_title("Prompt (m1_description)")
        ax2.axis("off")

        ax3 = plt.subplot(2, 2, 3)
        ax3.imshow(gen_img_pil)
        ax3.set_title("Generated")
        ax3.axis("off")

        ax4 = plt.subplot(2, 2, 4)
        ax4.imshow(gt_img_pil)
        ax4.set_title("Ground Truth (modality_1)")
        ax4.axis("off")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()


