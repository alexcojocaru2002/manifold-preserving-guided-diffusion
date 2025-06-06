import torch

from losses.loss import GuidanceLoss
import torch.nn.functional as F


class MSEGuidanceLoss(GuidanceLoss):
    def __init__(self, target: torch.Tensor):
        self.target = target

    def __call__(self, prediction: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(prediction, self.target, reduction="none")
        return loss.view(loss.size(0), -1).mean(dim=1)  # per-sample loss
