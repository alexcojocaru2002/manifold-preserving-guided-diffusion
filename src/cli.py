import os
import click

from pipelines.pipeline import StableDiffusionGenerator

@click.group()
def cli():
    """Run different scripts based on the provided arguments."""
    pass

@cli.command()
@click.option('--num_samples', type=int, required=True, help='Number of samples to visualize')
@click.option('--prompt', type=str, default="a cat", help='Text prompt for image generation')
def generate(num_samples: int, prompt: str):

    generator = StableDiffusionGenerator()

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