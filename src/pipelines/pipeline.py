import os
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler


class StableDiffusionGenerator:
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        # Load models and components
        self._load_models()

    def _load_models(self):
        print("Loading models...")

        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet").to(self.device)

        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000
        )

    def generate(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=100,
        guidance_scale=7.5,
        seed=0,
        output_dir="data"
    ):
        torch.manual_seed(seed)
        batch_size = len(prompt)

        # Tokenize and encode text
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Unconditional embeddings
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=text_input.input_ids.shape[-1],
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prepare latent noise
        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=torch.manual_seed(seed),
        ).to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        # Denoising loop
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to image
        latents = latents / 0.18215
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in images]

        return pil_images