import os
import sys

import torch

from src.pipelines import MPGDStableDiffusionGenerator


# def visualize_data(generator, prompt, num_samples):


def run(num_samples=1, prompt="", reference_image_path='', output_dir='visualizations' ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print(f'Running on {torch.cuda.get_device_name(0)}')

    generator = MPGDStableDiffusionGenerator(
        reference_image_path=reference_image_path
    )

    # Generate images
    print("Showing image")
    prompt = [prompt] * num_samples
    images = generator.generate(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42
    )
    for i, image in enumerate(images):
        print("Saving image " + str(i))
        image.save("data/image_" + str(i) + ".png")
