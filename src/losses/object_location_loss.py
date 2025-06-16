import json

import numpy
import torch
from typing import Dict, List

from PIL import ImageFont, Image, ImageDraw
from matplotlib import pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

from src.losses.loss import GuidanceLoss


class ObjectLocationLoss(GuidanceLoss):
    def __init__(self, reference_path: str, device: torch.device, image_key: str='image'):
        super().__init__()
        self.device = device
        self.reference = self._load_reference(reference_path, image_key)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.frcnn = fasterrcnn_resnet50_fpn(weights=weights).to(device)
        self.count = 0
        self.frcnn.train()
        for param in self.frcnn.parameters():
            param.requires_grad = False
        for m in self.frcnn.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        self.COCO_CLASSES = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
        self.lambda_tv = 1e-5


    def _load_reference(self, path: str, key: str) -> List[Dict[str, torch.Tensor]]:
        with open(path, "r") as f:
            data = json.load(f)

        boxes = torch.tensor(data[key]["boxes"], dtype=torch.float32, device=self.device)
        labels = torch.tensor(data[key]["labels"], dtype=torch.int64, device=self.device)

        return [{"boxes": boxes, "labels": labels}]

    def unnormalize(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
        return img * std + mean

    def total_variation_loss(self, image):
        return torch.sum(torch.abs(image[:, :, :-1] - image[:, :, 1:])) + \
            torch.sum(torch.abs(image[:, :-1, :] - image[:, 1:, :]))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute object-level guidance loss using Faster R-CNN.

        The total loss ℓ includes:
        (1) anchor classification loss ('loss_objectness'),
        (2) bounding box regression loss at the RPN ('loss_rpn_box_reg'),
        (3) region classification loss ('loss_classifier').

        Losses (1) and (2) are computed at the region proposal head.
        Loss (3) is computed at the region classification head.
        """
        normalized_image = image.to(self.device)

        normalized_image = (normalized_image / 2 + 0.5).clamp(0, 1)

        print(normalized_image.min().item(), normalized_image.max().item())
        loss_dict = self.frcnn(normalized_image, self.reference)

        total_loss = (
                loss_dict['loss_objectness'] +
                loss_dict['loss_rpn_box_reg'] +
                loss_dict['loss_classifier']
        )

        # Inference
        if self.count % 10 == 0:
            self.detect_and_show(normalized_image, threshold=0.5, label_color="red", width=3, font_size=16)
        self.count = self.count + 1
        return total_loss + self.lambda_tv * self.total_variation_loss(normalized_image)

    @torch.no_grad()                       # turn off grad – same effect as the with-block
    def detect_and_show(self, image, threshold=0.7, label_color="red", width=3, font_size=16):
        """
        Run Faster R-CNN on a 4D tensor [1,3,H,W] and display the result.

        Args
        ----
        frcnn      : a pretrained Faster R-CNN model
        image      : torch.Tensor on the *same* device as the model
        threshold  : confidence cutoff
        label_color: bbox / text colour recognised by Pillow
        width      : bbox line width
        font_size  : text size (draw_bounding_boxes handles fonts for us)

        Returns
        -------
        PIL.Image with the drawn boxes (also shown with .show()).
        """
        # Remove dummy batch dim -- easier to keep everything as tensors
        img = image.squeeze(0)                            # [3,H,W]

        # ---------- inference ----------
        self.frcnn.eval()
        # print(img.min().item(), img.max().item())
        with torch.no_grad():
            pred = self.frcnn([img])[0]                            # dict of tensors
        keep = pred["scores"] > threshold                 # boolean mask *still on GPU*
        boxes  = pred["boxes"][keep]
        labels = pred["labels"][keep]

        # ---------- drawing ----------
        # draw_bounding_boxes expects uint8, C-first
        # img = self.unnormalize(img)
        img_uint8 = (img * 255).clamp_(0, 255).to(torch.uint8)
        # Convert numeric labels to strings once; avoids Python loop over boxes
        label_strs = [self.COCO_CLASSES[int(l)] for l in labels.tolist()]
        print(boxes)
        print(label_strs)
        print(pred["scores"][keep])

        drawn = draw_bounding_boxes(
            img_uint8, boxes,
            labels=label_strs,
            colors=label_color,
            width=width,
            font_size=font_size,
        )

        pil_img = to_pil_image(drawn.cpu())               # single device→CPU hop
        # pil_img = to_pil_image(img_uint8)
        print("Showing image")

        # ---------- always display ----------
        plt.figure(figsize=(6, 6))  # minimal addition
        plt.imshow(pil_img)
        plt.axis("off")
        plt.show()
        self.frcnn.train()
        return pil_img