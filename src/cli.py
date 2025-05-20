import os
import click
import torch
from diffusers import StableDiffusionPipeline

@click.group()
def cli():
    """Run different scripts based on the provided arguments."""
    pass

@cli.command()
@click.option('--num_samples', type=int, required=True, help='Number of samples to visualize')
@click.option('--prompt', type=str, default="a cat", help='Text prompt for image generation')
def generate_images(num_samples: int, prompt: str):

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

    if torch.cuda.is_available():
        pipe.to("cuda")
        print(f'Running on {torch.cuda.get_device_name(0)}')

    # Generate images
    prompt = [prompt] * num_samples
    images = pipe(prompt).images

    # Save images
    os.makedirs('data', exist_ok=True)
    for i, img in enumerate(images):
        print(f'Generating image {i+1}/{num_samples}')
        img.save(f'data/image_{i}.png')
    
if __name__ == '__main__':
    cli()