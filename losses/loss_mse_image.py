from sympy.printing.pytorch import torch


def loss(y: torch.Tensor):

    def _loss(image: torch.Tensor):

        loss = torch.nn.functional.mse_loss(image, y, reduction="none")
        loss = loss.view(loss.size(0), -1).mean(dim=1)

        return loss

    return _loss