import os
import sys

import torch

from src.pipelines.pipeline import MPGDStableDiffusionGenerator


# def visualize_data(generator, prompt, num_samples):


def run(num_samples=1, reference_path=''):

    generator = MPGDStableDiffusionGenerator(
        reference_path=reference_path
    )

    # Generate images
    print("Showing image")
    images = generator.generate(
        batch_size=num_samples,
        height=512,
        width=512,
        num_inference_steps=50,
        seed=42
    )
    for i, image in enumerate(images):
        print("Saving image " + str(i))
        image.save("data/image_" + str(i) + ".png")
