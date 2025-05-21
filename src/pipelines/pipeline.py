import os
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler


class StableDiffusionGenerator:
    
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Running on {torch.cuda.get_device_name(0)}')
        self.model_id = model_id
        self._load_models()

    def _load_models(self):
        print("Loading models...")
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet").to(self.device)
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )

    def _encode_prompts(self, prompt_list: list[str]):
        text_input = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(
            [""] * len(prompt_list),
            padding="max_length",
            max_length=text_input.input_ids.shape[-1],
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        return torch.cat([uncond_embeddings, text_embeddings])

    def _generate_latents(self, batch_size: int, height: int, width: int, seed: int):
        torch.manual_seed(seed)
        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=torch.manual_seed(seed),
        ).to(self.device)
        return latents * self.scheduler.init_noise_sigma

    def _denoise_latents(self, 
        latents: torch.Tensor, 
        text_embeddings: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        ):
        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

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
        num_inference_steps: int=50,
        guidance_scale: float=7.5,
        seed: int=42,
    ):
        batch_size = len(prompt)

        text_embeddings = self._encode_prompts(prompt)
        latents = self._generate_latents(batch_size, height, width, seed)
        latents = self._denoise_latents(latents, text_embeddings, num_inference_steps, guidance_scale)
        return self._decode_latents(latents)