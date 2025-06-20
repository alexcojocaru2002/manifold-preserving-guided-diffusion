import json

import numpy
import torch
from typing import Dict, List

from PIL import ImageFont, Image, ImageDraw
from matplotlib import pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, \
    MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import pandas as pd

from src.losses.loss import GuidanceLoss


class ObjectLocationLoss(GuidanceLoss):
    """Combined Mask R-CNN + RetinaNet guidance with FP penalty, plus logging."""

    def __init__(
        self,
        reference_path: str,
        device: torch.device,
        image_key: str = "image",
        w_mask: float = 0.3,
        w_ret: float = 0.4,
        lambda_neg: float = 0.15,
        iou_thr: float = 0.4,
    ):
        super().__init__()
        self.device = device
        self.reference = self._load_reference(reference_path, image_key)
        self.w_mask = w_mask
        self.w_ret = w_ret
        self.lambda_neg = lambda_neg
        self.iou_thr = iou_thr

        # detectors
        self.mask_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.ret_weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        self.det_mask = maskrcnn_resnet50_fpn(weights=self.mask_weights).to(device)
        self.det_ret = retinanet_resnet50_fpn(weights=self.ret_weights).to(device)

        # freeze
        for det in (self.det_mask, self.det_ret):
            det.train()
            for p in det.parameters():
                p.requires_grad = False
            for m in det.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        self.COCO_CLASSES = self.mask_weights.meta["categories"]
        self.count = 0
        self.loss_log = []  # For tracking per-step loss info

    def _load_reference(self, path: str, key: str) -> List[Dict[str, torch.Tensor]]:
        with open(path, "r") as f:
            data = json.load(f)
        boxes = torch.tensor(data[key]["boxes"], dtype=torch.float32, device=self.device)
        labels = torch.tensor(data[key]["labels"], dtype=torch.int64, device=self.device)
        return [{"boxes": boxes, "labels": labels}]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        norm_img = (image.to(self.device) / 2 + 0.5).clamp(0, 1)

        # Build masks for Mask R-CNN
        targets = []
        for _ in range(B):
            t = {k: v.clone() for k, v in self.reference[0].items()}
            if len(t["boxes"]):
                masks = torch.zeros((len(t["boxes"]), H, W), dtype=torch.uint8, device=self.device)
                for i, (x1, y1, x2, y2) in enumerate(t["boxes"]):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    masks[i, y1:y2, x1:x2] = 1
                t["masks"] = masks
            targets.append(t)

        # Mask R-CNN loss
        loss_mask = self.det_mask(norm_img, targets)
        L_mask = (
            loss_mask["loss_objectness"]
            + loss_mask["loss_rpn_box_reg"]
            + loss_mask["loss_classifier"]
            + loss_mask["loss_mask"]
        )

        # RetinaNet loss
        loss_ret = self.det_ret(norm_img, self.reference)
        L_ret = loss_ret["classification"] + loss_ret["bbox_regression"]

        # RetinaNet FP loss
        self.det_ret.eval()
        with torch.no_grad():
            preds = self.det_ret([norm_img.squeeze(0)])[0]
        self.det_ret.train()

        scores = preds["scores"]
        labels = preds["labels"]
        boxes_pred = preds["boxes"]
        gt_boxes = self.reference[0]["boxes"]

        if boxes_pred.shape[0]:
            ious = box_iou(boxes_pred, gt_boxes)
            max_iou = ious.max(dim=1).values
            fp_mask = max_iou < self.iou_thr
            valid_cls = torch.isin(labels, self.reference[0]["labels"])
            fp_mask |= ~valid_cls
            fp_loss = scores[fp_mask].mean() if fp_mask.any() else torch.zeros(1, device=self.device)
        else:
            fp_loss = torch.zeros(1, device=self.device)

        # Mask R-CNN FP loss
        self.det_mask.eval()
        with torch.no_grad():
            preds_m = self.det_mask([norm_img.squeeze(0)])[0]
        self.det_mask.train()

        scores_m = preds_m["scores"]
        labels_m = preds_m["labels"]
        boxes_m = preds_m["boxes"]

        if boxes_m.shape[0]:
            ious_m = box_iou(boxes_m, gt_boxes)
            max_iou_m = ious_m.max(dim=1).values
            fp_mask_m = (max_iou_m < self.iou_thr) | (~torch.isin(labels_m, self.reference[0]["labels"]))
            fp_loss_mask = scores_m[fp_mask_m].mean() if fp_mask_m.any() else torch.zeros(1, device=self.device)
        else:
            fp_loss_mask = torch.zeros(1, device=self.device)

        # Log losses
        self.loss_log.append({
            "step": self.count,
            "loss_mask": L_mask.item(),
            "loss_retina": L_ret.item(),
            "fp_loss_retina": fp_loss.item(),
            "fp_loss_maskrcnn": fp_loss_mask.item(),
        })

        # Visualize occasionally
        if self.count % 10 == 0:
            self.detect_and_show(norm_img, self.det_mask, threshold=0.7, color="lime",
                                 title=f"Mask R‑CNN t = {self.count}, threshold = 0.7")
            self.detect_and_show(norm_img, self.det_ret, threshold=0.7, color="cyan",
                                 title=f"RetinaNet t = {self.count}, threshold = 0.7")
            print("Saving Losses")
            self.get_loss_log()

        self.count += 1

        total = (
            self.w_mask * L_mask +
            self.w_ret * L_ret +
            self.lambda_neg * (fp_loss + fp_loss_mask)
        ) / (self.w_mask + self.w_ret + 2 * self.lambda_neg)

        return total

    def get_loss_log(self, filepath: str = "object_location_loss_log.csv") -> pd.DataFrame:
        df = pd.DataFrame(self.loss_log)
        df.to_csv(filepath, index=False)
        print(f"[ObjectLocationLoss] Loss log written to: {filepath}")
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