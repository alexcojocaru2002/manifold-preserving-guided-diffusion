import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from scheduler import MPGDLatentScheduler, AutoencoderKL, UNet2DConditionModel
from tqdm.auto import tqdm

class MPGDLatent:
    def __init__(self, loss, unconditional=True, prompt=None, batch_size=1, scheduler=None, num_inference_steps=15, vae=None, unet=None, tokenizer=None, text_encoder=None, height=512, width=512, torch_device='cuda'):

        self.loss = self._define_loss(loss)
        self.scheduler = self._define_scheduler(scheduler, num_inference_steps)
        self.vae = self._define_vae(vae)
        self.unet = self._define_unet(unet)
        self.tokenizer = self._define_tokenizer(tokenizer)
        self.text_encoder = self._define_text_encoder(text_encoder)

        self.generator = torch.manual_seed(42)
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.unconditional = unconditional
        self.torch_device = torch_device

        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        # )
        # print(torch.cuda.is_available())
        # if torch.cuda.is_available():
        #     self.pipe.to("cuda")

    def _define_loss(self, loss):
        pass

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
            else AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        )

    def _define_unet(self, unet):
        unet = (
            unet
            if unet is not None
            else UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        )

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

        return text_encoder
    
    def __call__(self):
        
        # ! This is not finished, need to finish the loop. also incorporate loss function nicely
        if self.unconditional:
            uncond_input = self.tokenizer([""] * self.batch_size, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]
            
            latents = torch.randn(
                (self.batch_size, self.unet.in_channels, self.height // 8, self.width // 8),
                generator=self.generator,
            )
            latents = latents * self.scheduler.init_noise_sigma
        
            for t in tqdm(self.scheduler.timesteps):
            



        # return image
