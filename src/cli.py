import os
import click

from pipelines.pipeline import MPGDStableDiffusionGenerator

@click.group()
def cli():
    """Run different scripts based on the provided arguments."""
    pass

@cli.command()
@click.option('-ns', '--num_samples', type=int, required=True, help='Number of samples to visualize')
@click.option('-p', '--prompt', type=str, default="a cat", help='Text prompt for image generation')
@click.option('-rip', '--reference_image_path', type=str, required=True, help='Path to the reference image')
def generate(
    num_samples: int, 
    prompt: str,
    reference_image_path: str,
    ):

    # Init MPGDStableDiffusionGenerator
    generator = MPGDStableDiffusionGenerator(
        reference_image_path=reference_image_path
    )

    # Generate images
    prompt = [prompt] * num_samples
    images = generator.generate(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=0
    )

    # Save images
    os.makedirs('data', exist_ok=True)
    for i, img in enumerate(images):
        print(f'Generating image {i+1}/{num_samples}')
        img.save(f'data/image_{i}.png')
    
if __name__ == '__main__':
    cli()