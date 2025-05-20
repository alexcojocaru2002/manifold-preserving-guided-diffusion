import torch
from diffusers import StableDiffusionPipeline

class OurModel():
    def __init__(self):
        super().__init__()
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        print(torch.cuda.is_available())
        if torch.cuda.is_available():
            self.pipe.to("cuda")

    def __call__(self, prompt):
        image = self.pipe(prompt).images[0]
        return image