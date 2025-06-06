import os
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from torchvision import transforms

from src.losses.loss import GuidanceLoss
from src.losses.loss_mse_image import MSEGuidanceLoss
from src.schedulers.mpgd_latent_scheduler import MPGDLatentScheduler

class MPGDStableDiffusionGenerator:

    def __init__(
            self,
            model_id:str = "CompVis/stable-diffusion-v1-4",
            loss: GuidanceLoss = MSEGuidanceLoss,
            memory_efficient: bool = False,
            seed: int = 42
            ):

        # Load image reference]
        # Changed name to reference for future updates which will guide the process by text or other format
        # if not reference_path:
        #     raise ValueError("Reference image path must be provided.")
        # if not os.path.exists(reference_path):
        #     raise FileNotFoundError(f"Reference image path '{reference_path}' does not exist.")
        #
        #
        # print(reference_path)

        self.memory_efficient = memory_efficient

        # Get device
        self.loss = loss
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Running on {torch.cuda.get_device_name(0)}')

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
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet").to(self.device)
        self.scheduler = MPGDLatentScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler", eta=0.0)

        if self.memory_efficient:
            self.vae.enable_slicing() 
            self.vae.enable_gradient_checkpointing()
            self.unet.enable_xformers_memory_efficient_attention()
            
    def _encode_prompts(self, batch_size: int) -> torch.Tensor:

        # # Get text embeddings for the prompt
        # text_input = self.tokenizer(
        #     prompt_list,
        #     padding="max_length",
        #     max_length=self.tokenizer.model_max_length,
        #     truncation=True,
        #     return_tensors="pt"
        # )
        # text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Get unconditioned embeddings (empty prompt)
        uncond_input = self.tokenizer(
            [""] * batch_size,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0].to(self.device)

        # Delete text encoder to free memory
        del self.text_encoder
        torch.cuda.empty_cache()

        return uncond_embeddings

    def _generate_latents(self, batch_size: int, height: int, width: int) -> torch.Tensor:

        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=self.generator,
        ).to(self.device)
        return latents * self.scheduler.init_noise_sigma

    def _denoise_latents(self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_inference_steps: int,
        ) -> torch.Tensor:

        self.scheduler.set_timesteps(num_inference_steps)
        i = 0
        for t in tqdm(self.scheduler.timesteps):

            latents = self.scheduler.scale_model_input(latents, timestep=t)

            with torch.inference_mode():
                noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample

            latents = self.scheduler.step(noise_pred, t, latents, loss=self.loss, vae=self.vae).prev_sample
            i = i + 1
            img = self._decode_latents(latents)
            img[0].save("data/image_" + str(i) + ".png")

        return latents

    def _decode_latents(self, latents: torch.Tensor):
        latents = latents / 0.18215
        with torch.inference_mode():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return [Image.fromarray(img) for img in images]

    def generate(
        self,
        batch_size: int,
        height: int=512,
        width: int=512,
        seed: int=42,
        num_inference_steps: int = 50,
    ):
        batch_size = batch_size
        # self.reference_embedding = self._get_image_embedding(height, width)
        text_embeddings = self._encode_prompts(batch_size)
        latents = self._generate_latents(batch_size, height, width)
        latents = self._denoise_latents(latents, text_embeddings, num_inference_steps)
        return self._decode_latents(latents)