import clip
import torch
import torchvision
import torch.nn.functional as F

from losses import GuidanceLoss


class CLIPImageGuidanceLoss(GuidanceLoss):
    def __init__(self, target: torch.Tensor, device: torch.device):
        model_name = "ViT-B/16"
        print(clip.__file__)  # The GitHub version may not have this
        self.model, _ = clip.load(model_name, device=device)
        self.clip_model = self.model.eval().requires_grad_(False)


        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466 * 2 - 1, 0.4578275 * 2 - 1, 0.40821073 * 2 - 1),
            (0.26862954 * 2, 0.26130258 * 2, 0.27577711 * 2)
        )
        self.im1 = F.interpolate(target, size=(224, 224), mode='bicubic')
        self.target = self.preprocess(self.im1)
        self.image_ref_features = self.clip_model.encode_image(self.target)

    def __call__(self, prediction: torch.Tensor) -> torch.Tensor:
        im1 = torch.nn.functional.interpolate(prediction, size=(224, 224), mode='bicubic')
        im1 = self.preprocess(im1)

        image_features = self.clip_model.encode_image(im1)

        gram1 = torch.mm(image_features.t(), image_features)
        gram2 = torch.mm(self.image_ref_features.t(), self.image_ref_features)

        diff = gram1 - gram2
        norm = torch.linalg.norm(diff)

        return norm
