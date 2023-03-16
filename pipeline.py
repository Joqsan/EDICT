
from transformers import CLIPTextModel, CLIPTokenizer
from scheduling_edict import EDICTScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
import torch
import tqdm

from typing import Union
import PIL
from PIL import Image
import numpy as np

model_path_clip = "openai/clip-vit-large-patch14"
model_path_diffusion = "CompVis/stable-diffusion-v1-4"

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=Image.Resampling.LANCZOS))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images
        
class Pipeline:
    def __init__(self, device) -> None:

        self.device = device

        with open("hf_auth", "r") as f:
            auth_token = f.readlines()[0].strip()
        model_path_diffusion = "CompVis/stable-diffusion-v1-4"

        self.unet = UNet2DConditionModel.from_pretrained(
            model_path_diffusion,
            subfolder="unet",
            use_auth_token=auth_token,
            revision="fp16",
            torch_dtype=torch.float16,
        ).double().to(device)

        self.vae = AutoencoderKL.from_pretrained(
            model_path_diffusion,
            subfolder="vae",
            use_auth_token=auth_token,
            revision="fp16",
            torch_dtype=torch.float16,
        ).double().to(device)

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
        self.clip = CLIPTextModel.from_pretrained(model_path_clip, torch_dtype=torch.float16).double().to(device)
        self.scheduler = EDICTScheduler()

    def _encode_prompt(self, prompt, device, do_classifier_free_guidance):
        text_inputs = self.clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=self.clip_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.clip(text_input_ids.to(self.device))[0]

        prompt_embeds = prompt_embeds.to(dtype=self.clip.dtype, device=device)

        if do_classifier_free_guidance:
            uncond_tokens = [""]

            max_length = prompt_embeds.shape[1]
            uncond_input = self.clip_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_prompt_embeds = self.clip(
                uncond_input.input_ids.to(device)
            )[0]

            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
        
        return prompt_embeds

    def get_timesteps(self, num_inference_steps, strength, device):
        t_limit = num_inference_steps - int(num_inference_steps * strength)

        bwd_timesteps = self.scheduler.timesteps[t_limit:].repeat_interleave(2)  # T, T, ...
        fwd_timesteps = bwd_timesteps.flip(0)  # 0, 0, ....

        return fwd_timesteps, bwd_timesteps
    
    @torch.no_grad()
    def prepare_latents(self, image, text_emb, timesteps, guidance_scale, leapfrog_steps, device, generator):
        image = image.to(device=device, dtype=text_emb.dtype)
        init_latents = self.vae.encode(image).latent_dist.sample(generator)\
        
        init_latents = self.vae.config.scaling_factor * init_latents

        base, model_input = init_latents.clone(), init_latents.clone()

        do_mixing_now = True
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            # 1. Do mixing
            if do_mixing_now:
                base, model_input = self.scheduler.reverse_mixing_layer(base, model_input)

                # 2. Do swap after calling reverse_mixing_layer and before computing eq. (15.2)
                if leapfrog_steps:
                    base, model_input = model_input, base
            
            # 3. Compute Unet
            latent_model_input = torch.cat([model_input] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_emb
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # 4. Do noise step
            base, model_input = self.scheduler.reverse_step(sample=base, model_input=model_input, model_output=noise_pred, timestep=t)
            do_mixing_now ^= True

        if not leapfrog_steps:
            base, model_input = model_input, base

        return base, model_input

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.no_grad()
    def __call__(
            self,
            base_prompt,
            target_prompt,
            image,
            leapfrog_steps,
            strength,
            num_inference_steps,
            guidance_scale,
            device,
            generator
    ):
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        base_embeds = self._encode_prompt(
            base_prompt,
            device,
            do_classifier_free_guidance,
        )

        # 3. Encode input prompt
        target_embeds = self._encode_prompt(
            target_prompt,
            device,
            do_classifier_free_guidance,
        )

        # 4. Preprocess image
        image = preprocess(image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        fwd_timesteps, bwd_timesteps = self.get_timesteps(num_inference_steps, strength, device)

        base, model_input = self.prepare_latents(image, base_embeds, fwd_timesteps, guidance_scale, leapfrog_steps, device, generator)

        # Do denoising loop
        do_mixing_now = True
        for i, t in tqdm(enumerate(bwd_timesteps), total=len(bwd_timesteps)):

            # 1. Compute Unet
            latent_model_input = torch.cat([model_input] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=target_embeds
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 2. Do denoise step
            base, model_input = self.scheduler.forward_step(sample=base, model_input=model_input, model_output=noise_pred, timestep=t)
            do_mixing_now ^= True

            # 3. Do mixing
            if do_mixing_now:

                base, model_input = self.scheduler.forward_mixing_layer(base, model_input)
                # 4. Do swap after calling forward_mixing_layer and before computing (14.1) 
                # for the next iteration (that is, at the end of the current iteration).
                if leapfrog_steps:
                    base, model_input = model_input, base

        # base and model_input are close to each other
        final_latent = base


        image = self.decode_latents(final_latent)
        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image)

        return pil_image