import argparse
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Run Stable Diffusion inference.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to your local model folder"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    args = parser.parse_args()

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,  # use float16 if your GPU supports it
    )
    pipe = pipe.to("cuda")  # move to GPU

    # Run inference
    building_types = ["apartment", "house", "office", "school", "hospital", "hotel", "restaurant", "bar", 
                    "cafe", "church", "mosque", "synagogue", "temple", "warehouse", "factory", "industrial facility"]
    waffle_prompts = [f"A floor plan of a {building_type}" for building_type in building_types]
    prompts = ["a detailed architectural floorplan in the style of an architect's drawing",]
    prompts = prompts + waffle_prompts
    n_prompts = len(prompts)
    for i in range(args.num_examples):
        image = pipe(prompts[i % n_prompts]).images[0]
        prompt = prompts[i % n_prompts]
        
        # Create figure with title like in test_controlnet
        fig = plt.figure(figsize=(8, 10))
        ax = plt.subplot(1, 1, 1)
        ax.imshow(image)
        ax.set_title(prompt, wrap=True, fontsize=10)
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(f"outputs/waffle/waffle_{i}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization to: outputs/waffle/waffle_{i}.png")

if __name__ == "__main__":
    main()
