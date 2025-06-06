import torch
import torch.nn.functional as F
import lpips
import piq


class SSGuidanceLoss:
    def __init__(
        self,
        original_image,
        lpips_lambda=0.1,
        msss_lambda=0.1,
        spatial_lambda=0.1,
        device="cuda",
    ):
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

        self.lpips_lambda = lpips_lambda
        self.msss_lambda = msss_lambda
        self.spatial_lambda = spatial_lambda

    def lpips_loss(self, clean_image_estimation):
        return self.loss_fn_alex(self.reference, clean_image_estimation)

    def msss_loss(self, clean_image_estimation):
        # Ensure self.reference and clean_image_estimation are in range [0, 1]
        # Use torch.nn.functional.normalize to scale to [0, 1] per image
        ref = (self.reference - self.reference.min()) / (
            self.reference.max() - self.reference.min() + 1e-8
        )
        est = (clean_image_estimation - clean_image_estimation.min()) / (
            clean_image_estimation.max() - clean_image_estimation.min() + 1e-8
        )

        # MSSSIM expects (B, C, H, W) tensors in [0, 1]
        return 1.0 - piq.multi_scale_ssim(est, ref, data_range=1.0)

    def spatial_adaptive_loss(self, clean_image_estimation):
        # Assume self.reference and clean_image_estimation are torch tensors with shape (B, C, H, W)
        ref = self.reference
        est = clean_image_estimation

        # Compute gradients (Sobel filters)
        def compute_gradients(img):
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
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # (B, C, H, W)
            return grad_x, grad_y, grad_mag

        grad_ref_x, grad_ref_y, grad_ref_mag = compute_gradients(ref)
        grad_est_x, grad_est_y, grad_est_mag = compute_gradients(est)

        # Compute per-pixel weights based on reference gradient magnitude
        # Normalize weights to [0, 1] per image
        weights = grad_ref_mag
        weights = (weights - weights.amin(dim=(2, 3), keepdim=True)) / (
            weights.amax(dim=(2, 3), keepdim=True)
            - weights.amin(dim=(2, 3), keepdim=True)
            + 1e-8
        )

        # Compute weighted MSE loss between gradients
        loss_x = ((grad_ref_x - grad_est_x) ** 2 * weights).mean()
        loss_y = ((grad_ref_y - grad_est_y) ** 2 * weights).mean()
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
