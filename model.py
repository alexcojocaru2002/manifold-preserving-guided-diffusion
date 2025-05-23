import torch
from transformers import CLIPTextModel, CLIPTokenizer
from scheduler import MPGDLatentScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from tqdm.auto import tqdm


class MPGDLatent:
    def __init__(
        self,
        loss_fn,
        batch_size=1,
        scheduler=None,
        num_inference_steps=15,
        vae=None,
        unet=None,
        tokenizer=None,
        text_encoder=None,
        height=512,
        width=512,
        torch_device="cuda",
    ):

        self.torch_device = torch_device

        self.loss = loss_fn
        self.num_inference_steps = num_inference_steps

        self.scheduler = self._define_scheduler(scheduler, self.num_inference_steps)
        self.vae = self._define_vae(vae)
        self.unet = self._define_unet(unet)
        self.tokenizer = self._define_tokenizer(tokenizer)
        self.text_encoder = self._define_text_encoder(text_encoder)

        self.generator = torch.manual_seed(42)
        self.height = height
        self.width = width
        self.batch_size = batch_size

    def _define_scheduler(self, scheduler, num_inference_steps):
        scheduler = (
            scheduler
            if scheduler is not None
            else MPGDLatentScheduler.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="scheduler", eta=0.0
            )
        )
        scheduler.set_timesteps(num_inference_steps)

        return scheduler

    def _define_vae(self, vae):
        vae = (
            vae
            if vae is not None
            else AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="vae"
            )
        )
        vae.to(self.torch_device)

        return vae

    def _define_unet(self, unet):
        unet = (
            unet
            if unet is not None
            else UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="unet"
            )
        )
        unet.to(self.torch_device)

        return unet

    def _define_tokenizer(self, tokenizer):
        tokenizer = (
            tokenizer
            if tokenizer is not None
            else CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        )

        return tokenizer

    def _define_text_encoder(self, text_encoder):
        text_encoder = (
            text_encoder
            if text_encoder is not None
            else CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        )
        text_encoder.to(self.torch_device)

        return text_encoder

    def __call__(self):
        uncond_input = self.tokenizer([""] * self.batch_size, return_tensors="pt")
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.torch_device)
        )[0]

        latents = torch.randn(
            (self.batch_size, self.unet.in_channels, self.height // 8, self.width // 8),
            generator=self.generator,
            # device=self.torch_device,
        )
        latents = latents.to(self.torch_device)
        latents = latents * self.scheduler.init_noise_sigma

        for t in tqdm(self.scheduler.timesteps):
            latents = self.scheduler.scale_model_input(latents, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=uncond_embeddings.to(self.torch_device),
                ).sample

            # compute the previous noisy sample z_t -> z_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, loss=self.loss
            ).prev_sample

        # scale and decode the image latents with vae
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        latents = latents / scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return image
