import torch
from torch.amp import autocast
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from torchvision import transforms

from losses.loss import GuidanceLoss
from losses.loss_mse_image import MSEGuidanceLoss
from schedulers.mpgd_latent_scheduler import MPGDLatentScheduler

class MPGDStableDiffusionGenerator:

    def __init__(
            self,
            model_id:str = "CompVis/stable-diffusion-v1-4",
            loss: GuidanceLoss = MSEGuidanceLoss,
            memory_efficient: bool = False,
            use_fp16: bool = False,
            seed: int = 42,
            guidance_scale: float = 7.5,
            loss_guidance_scale: float = 20.0
            ):

        self.loss_guidance_scale = loss_guidance_scale
        self.guidance_scale = guidance_scale
        self.use_fp16 = use_fp16
        self.memory_efficient = memory_efficient

        # Get device
        self.loss = loss
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize models
        self.model_id = model_id
        self._load_models()
        self.generator = torch.manual_seed(seed)

    def _get_image_embedding(self, image: Image.Image, height: int = 512, width: int = 512) -> torch.Tensor:
        # Resize image to target size
        image = image.resize((width, height))

        # Preprocess: convert to tensor normalized in [-1, 1]
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device, dtype=self.vae.dtype)  # shape [1,3,H,W]
        image_tensor = 2.0 * image_tensor - 1.0  # scale from [0,1] to [-1,1]

        return image_tensor

    def _load_models(self):
        print("Loading models...")
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.vae = AutoencoderKL.from_pretrained(self.model_id, torch_dtype=dtype, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, torch_dtype=dtype, subfolder="unet").to(self.device)
        self.scheduler = MPGDLatentScheduler.from_pretrained(self.model_id, subfolder="scheduler", eta=0.0)

        if self.memory_efficient:
            self.vae.enable_slicing() 
            self.vae.enable_gradient_checkpointing()
            self.unet.enable_gradient_checkpointing()
            self.unet.enable_xformers_memory_efficient_attention()
            
    def _encode_prompts(self,
        prompt: str,
        batch_size: int
        ) -> torch.Tensor:

        # Get text embeddings for the prompt
        prompt_list = [prompt] * batch_size
        text_input = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids)[0]

        # Get unconditioned embeddings (empty prompt)
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding = "max_length",
            max_length = max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]

        # Cat embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(self.device)

        return text_embeddings

    def _generate_latents(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=self.generator,
        ).to(self.device)
        return latents * self.scheduler.init_noise_sigma

    def _denoise_latents(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:

        self.scheduler.set_timesteps(num_inference_steps)
        use_fp16 = self.unet.dtype == torch.float16
        for t in tqdm(self.scheduler.timesteps):

            # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # Scale
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # Use autocast
            with torch.inference_mode(), autocast(device_type=self.device, dtype=torch.float16, enabled=use_fp16):
                noise_pred = self.unet(
                    latent_model_input.to(self.unet.dtype), 
                    t, 
                    encoder_hidden_states=text_embeddings.to(self.unet.dtype)
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # MPGD update
            latents = self.scheduler.step(
                noise_pred, 
                t, 
                latents, 
                loss=self.loss, 
                vae=self.vae,
                lr_scale=self.loss_guidance_scale
            ).prev_sample

        return latents

    def _decode_latents(self, latents: torch.Tensor):
        latents = latents / 0.18215
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return [Image.fromarray(img) for img in images]

    def generate(
        self,
        prompt: str,
        batch_size: int,
        height: int=512,
        width: int=512,
        num_inference_steps: int = 50,
    ):
        text_embeddings = self._encode_prompts(prompt, batch_size)
        latents = self._generate_latents(batch_size, height, width)
        latents = self._denoise_latents(latents, text_embeddings, num_inference_steps)
        return self._decode_latents(latents)