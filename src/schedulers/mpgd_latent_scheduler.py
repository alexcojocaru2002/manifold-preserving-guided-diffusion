import torch
from diffusers import DDIMScheduler, AutoencoderKL
from typing import Optional, Tuple, Union
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

from src.losses import GuidanceLoss


class MPGDLatentScheduler(DDIMScheduler):

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        loss: GuidanceLoss,
        vae: AutoencoderKL,
        lr_scale: float = 100000,
        eta: float = 0.0
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`]
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. Get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. Compute alphas, betas
        # α_{t}
        alpha_prod_t = self.alphas_cumprod[timestep]

        # α_{t-1}
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        # 1-α_{t}
        beta_prod_t = 1 - alpha_prod_t

        # 3. Compute z0_t [line 4 of MPGD]
        pred_original_latent_sample = (
            (1 / alpha_prod_t ** (0.5)) * (sample - beta_prod_t ** (0.5) * model_output)
        ).requires_grad_(True)

        # Calculate loss here. Loss function is given to the scheduler, should be defined outside

        # Scale and decode image latents with vae so we can use mse loss
        # took the vae out of the mse function because gradient issues make it not guide properly
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        latents = pred_original_latent_sample / scaling_factor
        image = vae.decode(latents, return_dict=False)[0]
        loss_f = loss(image)

        loss_gradient = torch.autograd.grad(
            loss_f,
            pred_original_latent_sample,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
            is_grads_batched=False,
        )[0]

        print("∇ loss norm:", loss_gradient.norm().item())

        # ! c_t formula is from their implementation, but results in very small gradients
        # c_t = 0.0075 / alpha_prod_t.sqrt()
        c_t = lr_scale * 0.0075 / alpha_prod_t.sqrt()

        # 4. Steer z0_t towards our guidance objective [line 5 of MPGD]
        pred_original_latent_sample = pred_original_latent_sample - c_t * loss_gradient

        # 5. Compute variance: "sigma_t(η)"
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        pred_epsilon = model_output

        # 6. Compute "direction pointing to z_t"
        # This is line 7 from MPGD
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_epsilon

        # 7. Compute z_{t-1}
        # This is line 6 from MPGD
        prev_latent_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_latent_sample
            + pred_sample_direction
        )

        prev_latent_sample = prev_latent_sample.to(model_output.dtype)
        pred_original_latent_sample = pred_original_latent_sample.to(model_output.dtype)

        return DDIMSchedulerOutput(
            prev_sample=prev_latent_sample,
            pred_original_sample=pred_original_latent_sample,
        )