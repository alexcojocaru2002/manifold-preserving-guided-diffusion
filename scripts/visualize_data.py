import os
import sys

import torch

from model import OurModel

def visualize_data(model, prompt, num_samples):
    print("Showing image")
    for i in range(num_samples):
        image = model(prompt, 50)
        image.save("data/image_" + str(i) + ".png")
        print("Image saved in " + "data/image_" + str(i) + ".png")

def run(num_samples=1, prompt="a cat", output_dir='visualizations' ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print(f'Running on {torch.cuda.get_device_name(0)}')

    model = OurModel()
    visualize_data(model, prompt, num_samples)
