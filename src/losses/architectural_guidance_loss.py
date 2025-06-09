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
        arch_alpha: float = 0.5,
        struct_alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        architectural_style_prompt = f"Ensure that the desing style of the architectural structure showed matches the description: {prompt}"
        structural_accuracy_prompt = f"Ensure that the architectural elements are appropiately sized and proportioned to have structural realism and design"

        # Load and freeze CLIP model on the given device
        model, _ = clip.load("ViT-B/16", device=device, jit=False) # other model: "ViT-B/32",
        self.clip = model.eval().requires_grad_(False)

        # Tokenize and encode prompt to get text embedding
        with th.no_grad():

            # Architectural style embeddding
            architectural_style_tokens = clip.tokenize([architectural_style_prompt]).to(device)  # [1, 77]
            architectural_style_text_embed = model.encode_text(architectural_style_tokens)       # [1, D]
            architectural_style_text_embed = architectural_style_text_embed / architectural_style_text_embed.norm(dim=-1, keepdim=True)  # normalize

            # Structural accuracy embeddding
            structural_accuracy_tokens = clip.tokenize([structural_accuracy_prompt]).to(device)  # [1, 77]
            structural_accuracy_text_embed = model.encode_text(structural_accuracy_tokens)       # [1, D]
            structural_accuracy_text_embed = structural_accuracy_text_embed / structural_accuracy_text_embed.norm(dim=-1, keepdim=True)  # normalize

        # Final embeddings
        self.architectural_style_text_embed = architectural_style_text_embed.detach()  # [1, D]
        self.structural_accuracy_text_embed = structural_accuracy_text_embed.detach()  # [1, D]

        # Normalization
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

        self.arch_alpha = arch_alpha
        self.struct_alpha = struct_alpha

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
        x = self.normalize(x)

        # Image embedding
        img_embed = self.clip.encode_image(x)                                            # [B, D]

        # Architectural style loss
        txt_embed = self.architectural_style_text_embed.expand_as(img_embed)             # [B, D]
        arch_loss = 1.0 - F.cosine_similarity(img_embed, txt_embed, dim=-1)              # [B]

        # Structural accuracy loss                                         
        txt_embed = self.structural_accuracy_text_embed.expand_as(img_embed)             # [B, D]
        struct_loss = 1.0 - F.cosine_similarity(img_embed, txt_embed, dim=-1)            # [B]

        return self.arch_alpha * arch_loss + self.struct_alpha * struct_loss