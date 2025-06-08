import clip
import torch as th
import torch.nn.functional as F
from torchvision import transforms

from losses.loss import GuidanceLoss

class ArchitecturalGuidanceLoss(GuidanceLoss):
    """
    Lightweight semantic guidance loss for MPGD-LDM.

    Internally loads CLIP (ViT-B/32) for semantic alignment using text prompt.

    Expects `image` decoded by AutoencoderKL.decode(...) in range [-1,1].
    """

    def __init__(
        self,
        device: th.device,
        prompt: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Load and freeze CLIP model on the given device
        model, _ = clip.load("ViT-B/16", device=device, jit=False) # other model: "ViT-B/16",
        self.clip = model.eval().requires_grad_(False)

        # Tokenize and encode prompt to get text embedding
        with th.no_grad():
            tokens = clip.tokenize([prompt]).to(device)  # [1, 77]
            text_embed = model.encode_text(tokens)       # [1, D]
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)  # normalize
        self.text_embed = text_embed.detach()  # [1, D]

    def __call__(self, image: th.Tensor) -> th.Tensor:
        """
        Compute the semantic guidance loss.

        Args:
            image (Tensor[B, 3, H, W]): output of VAE.decode in [-1,1].

        Returns:
            Tensor[B]: unreduced per-sample cosine loss.
        """
        # Map to [0,1] for CLIP input
        x = (image / 2 + 0.5).clamp(0, 1)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        x = normalize(x)

        # Semantic loss: 1 - cosine similarity between CLIP(image) and CLIP(text)
        img_embed = self.clip.encode_image(x)                         # [B, D]
        txt_embed = self.text_embed.expand_as(img_embed)             # [B, D]
        L_sem = 1.0 - F.cosine_similarity(img_embed, txt_embed, dim=-1)  # [B]

        return L_sem