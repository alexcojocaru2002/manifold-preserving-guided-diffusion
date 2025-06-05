from abc import ABC, abstractmethod
import torch

class GuidanceLoss(ABC):
    @abstractmethod
    def __call__(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between the prediction and a target.

        Args:
            prediction (torch.Tensor): the predicted output (e.g., decoded image or latent)

        Returns:
            torch.Tensor: a tensor of loss values (per sample or scalar)
        """
        pass