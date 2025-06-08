import clip
import torch as th
import torch.nn.functional as F

from losses.loss import GuidanceLoss

class ArchitecturalGuidanceLoss(GuidanceLoss):
    """
    Lightweight classifier-based architectural guidance loss for MPGD-LDM.

    Internally loads CLIP (ViT-B/32) for semantic alignment.

    Expects `image` decoded by AutoencoderKL.decode(...) in range [-1,1].
    """

    def __init__(
        self,
        device: th.device,
        text_embed: th.Tensor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Load and freeze CLIP model on the given device
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.clip = model.eval().requires_grad_(False)

        # Load text embedding
        self.text_embed = text_embed.to(device).detach()      # [B, D]

    def compute(self, image: th.Tensor) -> th.Tensor:
        """
        Compute the combined guidance loss.

        Args:
            image (Tensor[B, 3, H, W]): output of VAE.decode in [-1,1].

        Returns:
            Tensor[B]: unreduced per-sample loss.
        """
        # Map to [0,1] for CLIP / classifier
        x = (image / 2 + 0.5).clamp(0, 1)

        # 1) Semantic loss: 1 - cos(CLIP_img, CLIP_text)
        img_embed = self.clip.encode_image(x)                   # [B, D]
        txt_embed = self.text_embed.expand_as(img_embed)       # [B, D]
        L_sem = 1.0 - th.nn.functional.cosine_similarity(img_embed, txt_embed, dim=-1)  # [B]

        return L_sem