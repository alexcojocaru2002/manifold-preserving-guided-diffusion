import os
import gc
import click
import torch

from PIL import Image
from torchvision import transforms
from losses.ss_loss import SSGuidanceLoss
from pipelines.pipeline import MPGDStableDiffusionGenerator
from losses.text_guidance_loss import CLIPTextGuidanceLoss
from losses.loss_mse_image import MSEGuidanceLoss
from transformers import CLIPModel, CLIPProcessor

from losses.clip_image_loss import CLIPImageGuidanceLoss
from losses.architectural_guidance_loss import ArchitecturalGuidanceLoss


@click.group()
def cli():
    """Run different scripts based on the provided arguments."""
    pass

@cli.command()
@click.option('-ns', '--num_samples', type=int, required=True, help='Number of samples to visualize')
@click.option('-rip', '--reference_image_path', type=str, required=True, help='Path to the reference image')
@click.option('-m', '--memory_efficient', is_flag=True, help='Use memory efficient mode')
@click.option('-s', '--seed', type=int, default=42, help='Random seed for reproducibility. Default is 42. Use -1 for random seed.')
def image_guidance_generator(
    num_samples: int,
    reference_image_path: str,
    memory_efficient: bool = False,
    seed: int = 42
    ):

    print("Memory efficient mode:", memory_efficient)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Running on {torch.cuda.get_device_name(0)}')

    # Prepare the reference image
    reference = Image.open(reference_image_path).convert("RGB")
    reference = reference.resize((512, 512))

    # Normaliz tensor in [-1, 1]
    image_tensor = transforms.ToTensor()(reference).unsqueeze(0).to(device)  # shape [1,3,H,W]
    image_tensor = 2.0 * image_tensor - 1.0  # scale from [0,1] to [-1,1]

    # Generate images
    generator = MPGDStableDiffusionGenerator(
        loss=CLIPImageGuidanceLoss(image_tensor, device=device),
        memory_efficient=memory_efficient,
        seed=seed
    )

    # Get random seed if it is not wanted reproducability
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()

    images = generator.generate(
        batch_size=num_samples,
        height=512,
        width=512,
        num_inference_steps=50,
    )
    for i, image in enumerate(images):
        print("Saving image " + str(i))
        image.save("data/image_" + str(i) + ".png")

@cli.command()
@click.option('-ns', '--num_samples', type=int, required=True, help='Number of samples to visualize')
@click.option('-p', '--prompt', type=str, required=True, help='Text prompt for image generation')
@click.option('-m', '--memory_efficient', is_flag=True, help='Use memory efficient mode')
@click.option('-s', '--seed', type=int, default=42, help='Random seed for reproducibility. Default is 42. Use -1 for random seed.')
def text_guidance_generator(
    num_samples: int,
    prompt: str,
    memory_efficient: bool = False,
    seed: int = 42
    ):

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Running on {torch.cuda.get_device_name(0)}')

    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


    # Generate images
    generator = MPGDStableDiffusionGenerator(
        loss=CLIPTextGuidanceLoss(prompt, clip_model, clip_processor, device=device),
        memory_efficient=memory_efficient,
        seed=seed
    )

    # Get random seed if it is not wanted reproducability
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()

    images = generator.generate(
        batch_size=num_samples,
        height=512,
        width=512,
        num_inference_steps=15,
    )
    for i, image in enumerate(images):
        print("Saving image " + str(i))
        image.save("data/image_" + str(i) + ".png")

@cli.command()
@click.option('-ns', '--num_samples', type=int, required=True, help='Number of samples to visualize')
@click.option('-p', '--prompt', type=str, required=True, help='Text prompt for image generation')
@click.option('-is', '--inference_steps', type=int, required=True, help='Number of inference steps.')
@click.option('-m', '--memory_efficient', is_flag=True, help='Use memory efficient mode')
@click.option('-s', '--seed', type=int, default=42, help='Random seed for reproducibility. Default is 42. Use -1 for random seed.')
@click.option('-fp16', '--use_fp16',  is_flag=True, help='Load VAE + UNet in float16')
@click.option('-gs', 'guidance_scale', type=float, default=20.0, help='Loss guidance scale. Reccomended values between 15 and 30')
def architectural_guidance_generator(
    num_samples: int,
    prompt: str,
    inference_steps: int,
    memory_efficient: bool = False,
    seed: int = 42,
    use_fp16: bool = False,
    guidance_scale: float = 20.0
    ):

    # Force garbage collection and clear CUDA memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Running on {torch.cuda.get_device_name(0)}')

    # Generate images
    generator = MPGDStableDiffusionGenerator(
        model_id="runwayml/stable-diffusion-v1-5",
        loss=ArchitecturalGuidanceLoss(
            prompt=prompt,
            device=device,
        ),
        memory_efficient=memory_efficient,
        use_fp16=use_fp16, 
        seed=seed,
        loss_guidance_scale=guidance_scale
    )

    # Get random seed if it is not wanted reproducability
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()

    images = generator.generate(
        prompt=prompt,
        batch_size=num_samples,
        height=512,
        width=512,
        num_inference_steps=inference_steps,
    )
    for i, image in enumerate(images):
        print("Saving image " + str(i))
        image.save("data/image_" + str(i) + ".png")

# Add comands to cli
cli.add_command(image_guidance_generator)
cli.add_command(text_guidance_generator)
    
if __name__ == '__main__':
    cli()