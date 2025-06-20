import json
import torch
from typing import Dict, List
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
from torchvision.ops import box_iou
import pandas as pd
from src.losses.loss import GuidanceLoss


class ObjectLocationLossSimple(GuidanceLoss):
    """Simple object guidance using Faster R-CNN detection loss, with logging."""

    def __init__(self, reference_path: str, device: torch.device, image_key: str = "image"):
        super().__init__()
        self.device = device
        self.reference = self._load_reference(reference_path, image_key)

        # Load pretrained Faster R-CNN
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn(weights=self.weights).to(device)

        # Freeze all model parameters
        self.detector.train()
        for p in self.detector.parameters():
            p.requires_grad = False
        for m in self.detector.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        self.COCO_CLASSES = self.weights.meta["categories"]
        self.count = 0
        self.loss_log = []  # Store step-wise loss info

    def _load_reference(self, path: str, key: str) -> List[Dict[str, torch.Tensor]]:
        with open(path, "r") as f:
            data = json.load(f)
        boxes = torch.tensor(data[key]["boxes"], dtype=torch.float32, device=self.device)
        labels = torch.tensor(data[key]["labels"], dtype=torch.int64, device=self.device)
        return [{"boxes": boxes, "labels": labels}]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        norm_img = (image.to(self.device) / 2 + 0.5).clamp(0, 1)

        # Duplicate targets for batch size
        targets = [self.reference[0].copy() for _ in range(B)]

        # Run loss in train mode
        loss_dict = self.detector(norm_img, targets)
        loss = sum(loss_dict.values())

        # Log loss info
        self.loss_log.append({
            "step": self.count,
            "total_loss": loss.item(),
            **{k: v.item() for k, v in loss_dict.items()}
        })

        # Optional: visualize detections
        if self.count % 20 == 0:
            self.detect_and_show(norm_img, self.detector, threshold=0.7,
                                 color="lime", title=f"Faster R-CNN t = {self.count}, confidence threshold = 0.7")
            print("Saving Losses")
            self.get_loss_log()

        self.count += 1

        print(f"Faster R-CNN loss: {loss.item():.4f}")
        return loss

    def get_loss_log(self, filepath: str = "rcnn_loss_log.csv") -> pd.DataFrame:
        df = pd.DataFrame(self.loss_log)
        df.to_csv(filepath, index=False)
        print(f"[ObjectLocationLossSimple] Loss log written to: {filepath}")
        return df

    @torch.no_grad()
    def detect_and_show(self, img, model, threshold=0.7, color="red", width=3, font_size=12, title="Detection"):
        model.eval()
        preds = model([img.squeeze(0)])[0]
        keep = preds["scores"] > threshold
        boxes = preds["boxes"][keep]
        labels = preds["labels"][keep]
        lblstr = [self.COCO_CLASSES[int(l)] for l in labels.tolist()]

        img_uint8 = (img.squeeze(0) * 255).clamp_(0, 255).to(torch.uint8)
        drawn = draw_bounding_boxes(img_uint8, boxes, labels=lblstr,
                                    colors=color, width=width, font_size=font_size)
        plt.figure(figsize=(6, 6))
        plt.title(title)
        plt.imshow(F.to_pil_image(drawn.cpu()))
        plt.tight_layout(pad=0)
        plt.axis("off")
        plt.show()
        model.train()