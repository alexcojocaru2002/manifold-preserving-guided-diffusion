from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
import torch
import os
from PIL import Image
from tqdm.auto import tqdm
from typing import Optional, Tuple, Union
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.schedulers.scheduling_ddim import randn_tensor

from losses.ss_loss import SSGuidanceLoss
from model import MPGDLatent
from scheduler import MPGDLatentScheduler
from scripts.visualize_data import visualize_image


######################################################################################################
# Creating a demo image for the demo loss

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
)


from diffusers import DDIMScheduler, LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)
scheduler = DDIMScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler", eta=0.0
)

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

prompt = ["a photograph of an astronaut riding a horse"]

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion

num_inference_steps = 50  # Number of denoising steps

guidance_scale = 7.5  # Scale for classifier-free guidance

generator = torch.manual_seed(17)  # Seed generator to create the initial latent noise

batch_size = len(prompt)

text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)

# uncond_input = tokenizer([""] * batch_size, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
# text_embeddings = uncond_embeddings

latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)

scheduler.set_timesteps(num_inference_steps)

latents = latents * scheduler.init_noise_sigma

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    # latent_model_input = latents

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample


# scale and decode the image latents with vae
scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
latents = latents / scaling_factor
with torch.no_grad():
    image = vae.decode(latents).sample

y = image

visualize_image(image, f"reference.png")

del text_encoder
del unet

# Done creating the demo image for the demo loss
#################################################################################################


# # # ! Demo MSE loss. Everyone should change this to cater to their own guidance
# def loss(y):
#     def _loss(clean_image_latent_estimation):
#         loss = torch.nn.functional.mse_loss(
#             clean_image_latent_estimation, y, reduction="none"
#         )
#         loss = loss.view(loss.size(0), -1).mean(dim=1)

#         return loss

#     return _loss


# loss_func = loss(y)

# mpgd = MPGDLatent(loss_func)
# image = mpgd()
# visualize_image(image, "result.png")


# Try out with an actual loss
#################################################################################################
ss_loss = SSGuidanceLoss(y, device=torch_device)

original_image = ss_loss.original_image
low_quality_image = ss_loss.low_quality_image
reference_image = ss_loss.reference

visualize_image(original_image, f"ss_original_image.png")
visualize_image(low_quality_image, f"ss_low_quality_image.png")
visualize_image(reference_image, f"ss_reference_image.png")

mpgd = MPGDLatent(ss_loss, num_inference_steps=50)
image = mpgd()

visualize_image(image, f"ss_result.png")
