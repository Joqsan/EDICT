import math
from typing import Optional

import numpy as np
import torch

from my_diffusers.configuration_utils import ConfigMixin, register_to_config
from my_diffusers.schedulers.scheduling_utils import SchedulerMixin


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


class EDICTScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        mix_weight=0.93,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[np.ndarray] = None,
        timestep_values: Optional[np.ndarray] = None,
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        tensor_format: str = "pt",
    ):
        if trained_betas is not None:
            self.betas = np.asarray(trained_betas)
        if beta_schedule == "linear":
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float64
            )
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}"
            )

        self.mix_weight = mix_weight
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this paratemer simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = (
            np.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

        # print(self.alphas.shape)

    def set_timesteps(self, num_inference_steps: int, device):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        

    def _get_alpha_and_beta(self, t):
        # want to run this for both current and previous timnestep
        t = int(t)

        alpha_prod = self.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod

        return alpha_prod, 1 - alpha_prod

    def forward_step(
        self,
        sample,
        model_input,
        model_output,
        timestep: int,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
        use_double=False,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )

        alpha_prod_t, beta_prod_t = self._get_alpha_and_beta(timestep)
        alpha_prod_t_prev, _ = self._get_alpha_and_beta(prev_timestep)

        alpha_quotient = (alpha_prod_t / alpha_prod_t_prev) ** 0.5
        first_term = (1.0 / alpha_quotient) * sample
        second_term = (1.0 / alpha_quotient) * (beta_prod_t**0.5) * model_output
        third_term = ((1 - alpha_prod_t_prev) ** 0.5) * model_output

        next_model_input = first_term - second_term + third_term
        return model_input, next_model_input

    def reverse_step(
        self,
        sample,
        model_input,
        model_output,
        timestep: int,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
        use_double=False,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        prev_timestep = (
            timestep - self.config.num_train_timesteps / self.num_inference_steps
        )

        alpha_prod_t, beta_prod_t = self._get_alpha_and_beta(timestep)
        alpha_prod_t_prev, _ = self._get_alpha_and_beta(prev_timestep)

        alpha_quotient = (alpha_prod_t / alpha_prod_t_prev) ** 0.5

        first_term = alpha_quotient * sample
        second_term = ((beta_prod_t) ** 0.5) * model_output
        third_term = alpha_quotient * ((1 - alpha_prod_t_prev) ** 0.5) * model_output

        next_model_input = first_term + second_term - third_term
        return model_input, next_model_input

    def reverse_mixing_layer(self, base, model_input):
        model_input = (model_input - (1 - self.mix_weight) * base) / self.mix_weight
        base = (base - (1 - self.mix_weight) * base) / self.mix_weight

        return [base, model_input]

    def forward_mixing_layer(self, base, model_input):
        base = self.mix_weight * base + (1 - self.mix_weight) * model_input
        model_input = (self.mix_weight) * model_input + (1 - self.mix_weight) * base
        return [base, model_input]
