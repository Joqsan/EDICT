import random
from difflib import SequenceMatcher

import torch
import numpy as np
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPTokenizer

# Have diffusers with hardcoded double-casting instead of float
from my_diffusers import (AutoencoderKL, DDIMScheduler, UNet2DConditionModel)




# Build our CLIP model
model_path_clip = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
clip = clip_model.text_model


# Getting our HF Auth token
with open("hf_auth", "r") as f:
    auth_token = f.readlines()[0].strip()
model_path_diffusion = "CompVis/stable-diffusion-v1-4"
# Build our SD model
unet = UNet2DConditionModel.from_pretrained(
    model_path_diffusion,
    subfolder="unet",
    use_auth_token=auth_token,
    revision="fp16",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    model_path_diffusion,
    subfolder="vae",
    use_auth_token=auth_token,
    revision="fp16",
    torch_dtype=torch.float16,
)

# Push to devices w/ double precision
device = "cuda"
unet.double().to(device)
vae.double().to(device)
clip.double().to(device)
print("Loaded all models")


def EDICT_editing(
    im_path,
    base_prompt,
    edit_prompt,
    steps=50,
    mix_weight=0.93,
    init_image_strength=0.8,
    guidance_scale=3,
    leapfrog_steps=True,
):
    """
    Main call of our research, performs editing with either EDICT or DDIM

    Args:
        im_path: path to image to run on
        base_prompt: conditional prompt to deterministically noise with
        edit_prompt: desired text conditoining
        steps: ddim steps
        mix_weight: Weight of mixing layers.
            Higher means more consistent generations but divergence in inversion
            Lower means opposite
            This is fairly tuned and can get good results
        init_image_strength: Editing strength. Higher = more dramatic edit.
            Typically [0.6, 0.9] is good range.
            Definitely tunable per-image/maybe best results are at a different value
        guidance_scale: classifier-free guidance scale
            3 I've found is the best for both our method and basic DDIM inversion
            Higher can result in more distorted results
        run_baseline:
            VERY IMPORTANT
            True is EDICT, False is DDIM
    Output:
        PAIR of Images (tuple)
        If run_baseline=True then [0] will be edit and [1] will be original
        If run_baseline=False then they will be two nearly identical edited versions
    """
    # Resize/center crop to 512x512 (Can do higher res. if desired)
    orig_im = (
        load_im_into_format_from_path(im_path) if isinstance(im_path, str) else im_path
    )  # trust OK

    # compute latent pair (second one will be original latent if run_baseline=True)
    latents = noise(
        base_prompt,
        init_image=orig_im,
        init_image_strength=init_image_strength,
        steps=steps,
        p=mix_weight,
        guidance_scale=guidance_scale,
        leapfrog_steps=leapfrog_steps,
    )
    
    
    # Denoise intermediate state with new conditioning
    gen = denoise(
        target_prompt=edit_prompt,
        fixed_starting_latent=latents,
        init_image_strength=init_image_strength,
        steps=steps,
        p=mix_weight,
        guidance_scale=guidance_scale,
        leapfrog_steps=leapfrog_steps
    )

    return gen


def center_crop(im):
    width, height = im.size  # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_im_into_format_from_path(im_path):
    return center_crop(Image.open(im_path)).resize((512, 512))



####################################

#### HELPER FUNCTIONS FOR OUR METHOD #####


def get_alpha_and_beta(self, t):
    t = int(t)

    alpha_prod = self.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod

    return alpha_prod, 1 - alpha_prod


# A DDIM forward step function
def forward_step(
    self,
    base,
    model_input,
    model_output,
    timestep: int,
):
    
    prev_timestep = (
        timestep - self.config.num_train_timesteps / self.num_inference_steps
    )

    alpha_prod_t, beta_prod_t = get_alpha_and_beta(self, timestep)
    alpha_prod_t_prev, beta_prod_t_prev = get_alpha_and_beta(self, prev_timestep)

    a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
    b_t = - a_t * (beta_prod_t ** 0.5) + beta_prod_t_prev ** 0.5
    next_model_input = a_t * base + b_t * model_output

    return model_input, next_model_input


# A DDIM reverse step function, the inverse of above
def reverse_step(
    self,
    base,
    model_input,
    model_output,
    timestep: int,
):

    prev_timestep = (
        timestep - self.config.num_train_timesteps / self.num_inference_steps
    )

    alpha_prod_t, beta_prod_t = get_alpha_and_beta(self, timestep)
    alpha_prod_t_prev, beta_prod_t_prev = get_alpha_and_beta(self, prev_timestep)

    a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
    b_t = - a_t * (beta_prod_t ** 0.5) + beta_prod_t_prev ** 0.5

    next_model_input = (base - b_t * model_output) /  a_t

    return model_input, next_model_input




@torch.no_grad()
def latent_to_image(latent):
    image = vae.decode(latent.to(vae.dtype) / 0.18215).sample
    image = prep_image_for_return(image)
    return image


def prep_image_for_return(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image


#############################

##### MAIN EDICT FUNCTION #######
# Use EDICT_editing to perform calls

def image_to_latent(im, width, height, seed):
        generator = torch.cuda.manual_seed(seed)

        # Change size to multiple of 64 to prevent size mismatches inside model
        width = width - width % 64
        height = height - height % 64

        if isinstance(im, torch.Tensor):
            # assume it's the latent
            # used to avoid clipping new generation before inversion
            init_latent = im.to(device)
        else:
            # Resize and transpose for numpy b h w c -> torch b c h w
            im = im.resize((width, height), resample=Image.Resampling.LANCZOS)
            im = np.array(im).astype(np.float64) / 255.0 * 2.0 - 1.0
            # check if black and white
            if len(im.shape) < 3:
                im = np.stack(
                    [im for _ in range(3)], axis=2
                )  # putting at end b/c channels

            im = torch.from_numpy(im[np.newaxis, ...].transpose(0, 3, 1, 2))

            # If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
            if im.shape[1] > 3:
                im = im[:, :3] * im[:, 3:] + (1 - im[:, 3:])

            # Move image to GPU
            im = im.to(device)
            # Encode image
            init_latent = (
                vae.encode(im).latent_dist.sample(generator=generator) * 0.18215
            )
            return init_latent

def encode_prompt(prompt):
    # CLIP Text Embeddings
    null_prompt = ""
    tokens_unconditional = clip_tokenizer(
        null_prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    embedding_unconditional = clip(
        tokens_unconditional.input_ids.to(device)
    ).last_hidden_state

    tokens_conditional = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    embedding_conditional = clip(
        tokens_conditional.input_ids.to(device)
    ).last_hidden_state

    return torch.cat([embedding_unconditional, embedding_conditional])


def reverse_mixing_layer(x, y, p):
    # Reverse mixing layer
    y = (y - (1 - p) * x) / p
    x = (x - (1 - p) * y) / p

    return [x, y]

@torch.no_grad()
def noise(
    base_prompt="",
    guidance_scale=7.0,
    steps=50,
    seed=1,
    width=512,
    height=512,
    init_image=None,
    init_image_strength=1.0,
    leapfrog_steps=True,
    beta_schedule="scaled_linear",
    p=0.93,
):
    
    # can take either pair (output of generative process) or single image
    if isinstance(init_image, list):
        init_latent = [image_to_latent(im, width, height, seed) for im in init_image]
    else:
        init_latent = image_to_latent(init_image, width, height, seed)
        # this is t_start for forward, t_end for reverse
    
    t_limit = steps - int(steps * init_image_strength)

    latent = init_latent

    if isinstance(latent, list):  # initializing from pair of images
        latent_pair = latent
    else:  # initializing from noise
        latent_pair = [latent.clone(), latent.clone()]


    # Set inference timesteps to scheduler
    scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule=beta_schedule,
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
        )
    scheduler.set_timesteps(steps)

    prompt_embeds = encode_prompt(base_prompt)
    
    timesteps = scheduler.timesteps[t_limit:]
    timesteps = timesteps.flip(0)

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

        latent_pair = reverse_mixing_layer(x=latent_pair[0], y=latent_pair[1], p=p)        

        # alternate EDICT steps
        for latent_j in range(2):
            
            latent_i = latent_j ^ 1

            if leapfrog_steps:
                if i % 2 == 0:
                    latent_i, latent_j = latent_j, latent_i

            model_input = latent_pair[latent_j]
            base = latent_pair[latent_i]

            latent_model_input = torch.cat([model_input] * 2)

            # Predict the noise residual
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            base, model_input = reverse_step(self=scheduler, base=base, model_input=model_input, model_output=noise_pred, timestep=t)
            model_input = model_input.to(base.dtype)

            latent_pair[latent_i] = model_input

    
    results = [latent_pair]
    return results if len(results) > 1 else results[0]


def forward_mixing_layer(x, y, p):
    # Mixing layer (contraction) during generative process
    x = (p * x + (1 - p) * y)
    y = (p * y + (1 - p) * x)

    return [x, y]

@torch.no_grad()
def denoise(
    target_prompt="",
    guidance_scale=7.0,
    steps=50,
    init_image_strength=1.0,
    leapfrog_steps=True,
    fixed_starting_latent=None,
    beta_schedule="scaled_linear",
    p=0.93,
):


    t_limit = steps - int(steps * init_image_strength)
    
    latent_pair = fixed_starting_latent

    # Set inference timesteps to scheduler
    scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule=beta_schedule,
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
        )
    scheduler.set_timesteps(steps)

    prompt_embeds = encode_prompt(target_prompt)

    timesteps = scheduler.timesteps[t_limit:]

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

        # alternate EDICT steps
        for latent_i in range(2):
            
            latent_j = latent_i ^ 1

            if leapfrog_steps:
                if i % 2 == 1:
                    latent_i, latent_j = latent_j, latent_i

            model_input = latent_pair[latent_j]
            base = latent_pair[latent_i]

            latent_model_input = torch.cat([model_input] * 2)

            # Predict the noise residual
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            base, model_input = forward_step(self=scheduler, base=base, model_input=model_input, model_output=noise_pred, timestep=t)
            model_input = model_input.to(base.dtype)

            latent_pair[latent_i] = model_input

        latent_pair = forward_mixing_layer(x=latent_pair[0], y=latent_pair[1], p=p)
        

    # decode latents to iamges
    images = []
    for latent_i in range(2):
        latent = latent_pair[latent_i] / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample
        images.append(image)

    # Return images
    return_arr = []
    for image in images:
        image = prep_image_for_return(image)
        return_arr.append(image)
    results = [return_arr]
    return results if len(results) > 1 else results[0]
