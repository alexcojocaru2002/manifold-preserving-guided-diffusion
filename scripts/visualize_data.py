import os
import sys

import torch

from model import OurModel

def visualize_data(model, num_samples):
    prompt = "a cat playing fotball"
    print("Showing image")
    for i in range(num_samples):
        image = model(prompt)
        image.save("data/image_" + str(i) + ".png")

def run(num_samples=1, output_dir='visualizations' ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print(f'Running on {torch.cuda.get_device_name(0)}')

    model = OurModel()
    visualize_data(model, num_samples)
