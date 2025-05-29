import os
import sys

import torch

from models import OurModel
from PIL import Image
from pathlib import Path


def visualize_data(model, prompt, num_samples):
    print("Showing image")
    for i in range(num_samples):
        image = model(prompt)
        image.save("data/image_" + str(i) + ".png")


def run(num_samples=1, prompt="a cat", output_dir="visualizations"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        print(f"Running on {torch.cuda.get_device_name(0)}")

    model = OurModel()
    visualize_data(model, prompt, num_samples)


def visualize_image(image, save_name, save_folder="visualization"):
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    full_save_path = save_path / save_name

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(full_save_path)
