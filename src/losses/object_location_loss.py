import json

import numpy
import torch
from typing import Dict, List

from PIL import ImageFont, Image, ImageDraw
from matplotlib import pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from src.losses.loss import GuidanceLoss


class ObjectLocationLoss(GuidanceLoss):
    def __init__(self, reference_path: str, device: torch.device, image_key: str='image'):
        super().__init__()
        self.device = device
        self.reference = self._load_reference(reference_path, image_key)
        self.frcnn = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
        self.frcnn.train()

    def _load_reference(self, path: str, key: str) -> List[Dict[str, torch.Tensor]]:
        with open(path, "r") as f:
            data = json.load(f)

        boxes = torch.tensor(data[key]["boxes"], dtype=torch.float32, device=self.device)
        labels = torch.tensor(data[key]["labels"], dtype=torch.int64, device=self.device)

        return [{"boxes": boxes, "labels": labels}]

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
        self.frcnn.train()
        image = image.to(self.device)
        loss_dict = self.frcnn(image, self.reference)

        total_loss = (
                loss_dict['loss_objectness'] +
                loss_dict['loss_rpn_box_reg'] +
                loss_dict['loss_classifier']
        )

        # Inference
        self.frcnn.eval()
        image = image.squeeze(0)
        with torch.no_grad():
            predictions = self.frcnn([image])  # list of one image

        # Extract predictions
        pred = predictions[0]
        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]
        print(labels)
        #
        # # Filter by confidence threshold
        # threshold = 0.5
        # keep = scores > threshold
        # filtered_boxes = boxes[keep]
        # filtered_labels = labels[keep]
        # print(filtered_labels)
        # filtered_scores = scores[keep]
        #
        # # Get image dimensions (assumes image shape is [3, H, W])
        # _, H, W = image.shape
        #
        # # Create black image using PIL
        # # Optional: use a default font
        # try:
        #     font = ImageFont.truetype("arial.ttf", size=16)
        # except:
        #     font = ImageFont.load_default()
        #
        # # Display the image
        # image_np = image.permute(1, 2, 0).detach().cpu().numpy()
        # image_np = (image_np * 255).clip(0, 255).astype(numpy.uint8)
        # image_real = Image.fromarray(image_np)
        # # image2 = image.permute(1, 2, 0).detach().cpu().numpy()
        # draw = ImageDraw.Draw(image_real)
        # for box, label in zip(filtered_boxes, filtered_labels):
        #     box = box.cpu().tolist()
        #     draw.rectangle(box, outline="red", width=3)
        #     draw.text((box[0] + 3, box[1] + 3), str(label.item()), fill="red", font=font)
        #
        # plt.imshow(image_real)
        # plt.axis('off')  # optional, hides axis ticks
        # plt.show()

        return total_loss