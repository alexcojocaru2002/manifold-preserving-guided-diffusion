import os
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from torchvision import transforms

from src.schedulers.mpgd_latent_scheduler import MPGDLatentScheduler

class MPGDStableDiffusionGenerator:

    def __init__(
            self,
            model_id:str = "CompVis/stable-diffusion-v1-4",
            reference_image_path: str = None,
            ):

        # Load image reference
        if not reference_image_path:
            raise ValueError("Reference image path must be provided.")
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(f"Reference image path '{reference_image_path}' does not exist.")
        self.reference_image = Image.open(reference_image_path).convert("RGB")

        # Get device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Running on {torch.cuda.get_device_name(0)}')

        # Initialize models
        self.model_id = model_id
        self._load_models()

    def _get_image_embedding(self, image: Image.Image, height: int = 512, width: int = 512) -> torch.Tensor:
        # Resize image to target size
        image = image.resize((width, height))

        # Preprocess: convert to tensor normalized in [-1, 1]
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device, dtype=self.vae.dtype)  # shape [1,3,H,W]
        image_tensor = 2.0 * image_tensor - 1.0  # scale from [0,1] to [-1,1]

        return image_tensor

    def _loss(self, y: torch.Tensor):

        def _loss(clean_image_latent_estimation: torch.Tensor):

            # Scale and decode image latents with vae
            scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
            latents = clean_image_latent_estimation / scaling_factor
            image = self.vae.decode(latents).sample

            loss = torch.nn.functional.mse_loss(image, y, reduction="none")
            loss = loss.view(loss.size(0), -1).mean(dim=1)

            return loss

        return _loss

    def _load_models(self):
        print("Loading models...")
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae").to(self.device).half()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).half()
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet").to(self.device).half()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).half()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.scheduler = MPGDLatentScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler", eta=0.0)

    def _encode_prompts(self, prompt_list: list[str]) -> torch.Tensor:

        # Get text embeddings for the prompt
        text_input = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Get unconditioned embeddings (empty prompt)
        uncond_input = self.tokenizer(
            [""] * len(prompt_list),
            padding="max_length",
            max_length=text_input.input_ids.shape[-1],
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        return torch.cat([uncond_embeddings, text_embeddings])

    def _generate_latents(self, batch_size: int, height: int, width: int, seed: int) -> torch.Tensor:

        torch.manual_seed(seed)
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=torch.manual_seed(seed),
        ).to(self.device).half()
        return latents * self.scheduler.init_noise_sigma

    def _denoise_latents(self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        ) -> torch.Tensor:

        self.scheduler.set_timesteps(num_inference_steps)
        self.loss = self._loss(self.reference_image_embedding)

        for t in tqdm(self.scheduler.timesteps):

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, self.loss).prev_sample

        return latents

    def _decode_latents(self, latents: torch.Tensor):
        latents = latents / 0.18215
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return [Image.fromarray(img) for img in images]

    def generate(
        self,
        prompt: list[str],
        height: int=512,
        width: int=512,
        guidance_scale: float=7.5,
        seed: int=42,
        num_inference_steps: int = 50,
    ):
        batch_size = len(prompt)
        self.reference_image_embedding = self._get_image_embedding(self.reference_image, height, width)

        text_embeddings = self._encode_prompts(prompt)
        latents = self._generate_latents(batch_size, height, width, seed)
        latents = self._denoise_latents(latents, text_embeddings, num_inference_steps, guidance_scale)
        return self._decode_latents(latents)