import torch
import torch.nn.functional as F
import lpips

from losses.loss import GuidanceLoss


class SSGuidanceLoss(GuidanceLoss):
    def __init__(self, original_image, device="cuda"):
        self.original_image = original_image

        # Assume original_image is a torch tensor of shape (B, C, H, W)
        low_quality_image = F.interpolate(
            original_image,
            scale_factor=0.25,
            mode="bilinear",
            align_corners=False,
        )
        self.low_quality_image = low_quality_image

        # Upsample low_quality_image by a factor of 4 to get reference_image
        self.reference = F.interpolate(
            self.low_quality_image,
            scale_factor=4.0,
            mode="bilinear",
            align_corners=False,
        )
        # self.reference.to(device)

        self.loss_fn_alex = lpips.LPIPS(net="alex").to(device)

        self.lpips_lambda = 0.5
        self.msss_lambda = 0.0
        self.spatial_lambda = 0.5

    def lpips_loss(self, clean_image_estimation):
        return self.loss_fn_alex(self.reference, clean_image_estimation)

    # TODO: Implement this loss
    def msss_loss(self, clean_image_estimation):
        return 0.0

    # TODO: Implement gradient-based weighting per pixel
    def spatial_adaptive_loss(self, clean_image_estimation):
        # Assume self.reference and clean_image_estimation are torch tensors with shape (B, C, H, W)
        ref = self.reference
        est = clean_image_estimation

        # Compute gradients (Sobel filters)
        def compute_gradients(img):
            # If input has 3 channels, apply the filter to each channel separately
            B, C, H, W = img.shape
            sobel_x = (
                torch.tensor(
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    dtype=img.dtype,
                    device=img.device,
                )
                .view(1, 1, 3, 3)
                .repeat(C, 1, 1, 1)
            )
            sobel_y = (
                torch.tensor(
                    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                    dtype=img.dtype,
                    device=img.device,
                )
                .view(1, 1, 3, 3)
                .repeat(C, 1, 1, 1)
            )
            grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
            grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
            return grad_x, grad_y

        grad_ref_x, grad_ref_y = compute_gradients(ref)
        grad_est_x, grad_est_y = compute_gradients(est)

        # Compute L2 loss between gradients
        loss_x = F.mse_loss(grad_ref_x, grad_est_x)
        loss_y = F.mse_loss(grad_ref_y, grad_est_y)
        return loss_x + loss_y

    def __call__(self, clean_image_estimation):
        lpips = self.lpips_loss(clean_image_estimation)
        msss_loss = self.msss_loss(clean_image_estimation)
        spatial_adaptive_loss = self.spatial_adaptive_loss(clean_image_estimation)

        return (
            self.lpips_lambda * lpips
            + self.msss_lambda * msss_loss
            + self.spatial_lambda * spatial_adaptive_loss
        )
