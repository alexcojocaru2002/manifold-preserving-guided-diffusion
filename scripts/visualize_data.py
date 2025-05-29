import os
import sys

import torch

from src.pipelines.pipeline import MPGDStableDiffusionGenerator
from losses.loss_mse_image import loss
from PIL import Image
from torchvision import transforms

# def visualize_data(generator, prompt, num_samples):


def run(num_samples=1, reference_path=''):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f'Running on {torch.cuda.get_device_name(0)}')
    reference_path = reference_path
    reference = Image.open(reference_path).convert("RGB")
    reference = reference.resize((512, 512))
    # Preprocess: convert to tensor normalized in [-1, 1]
    image_tensor = transforms.ToTensor()(reference).unsqueeze(0).to(device)  # shape [1,3,H,W]
    image_tensor = 2.0 * image_tensor - 1.0  # scale from [0,1] to [-1,1]

    generator = MPGDStableDiffusionGenerator(
        loss=loss(y=image_tensor)
    )

    # Generate images
    print("Showing image")
    images = generator.generate(
        batch_size=num_samples,
        height=512,
        width=512,
        num_inference_steps=50,
        seed=42, # TO DO: Make this random later
    )
    for i, image in enumerate(images):
        print("Saving image " + str(i))
        image.save("data/image_" + str(i) + ".png")
