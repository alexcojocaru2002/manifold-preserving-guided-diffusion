import json
import torch
from typing import Dict, List
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

        image = image.to(self.device)
        loss_dict = self.frcnn(image, self.reference)
        return sum(loss for loss in loss_dict.values())