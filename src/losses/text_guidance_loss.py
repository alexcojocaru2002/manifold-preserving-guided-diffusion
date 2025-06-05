from PIL import Image
from typing import Callable
from torchvision import transforms
import torch
import torch.nn.functional as F
from losses.loss import GuidanceLoss

# Example of loss for text to image, currently not working just a prototype
# Maybe needs a mse loss to guide it towards the visual content as well
class CLIPTextGuidanceLoss(GuidanceLoss):
    def __init__(
        self,
        prompt: str,
        clip_model,
        clip_processor,
        device: str = "cuda"
    ):
        self.prompt = prompt
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Resize to 224x224 and normalize for CLIP
        print(image.shape)
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1] if needed

        # Get text features
        text_inputs = self.clip_processor(
            text=[self.prompt] * image.shape[0],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        text_embeds = self.clip_model.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # Get image features
        image_embeds = self.clip_model.get_image_features(image)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        # does cosine similarity here
        similarity = (image_embeds * text_embeds).sum(dim=-1)
        loss = 1 - similarity

        return loss.mean()