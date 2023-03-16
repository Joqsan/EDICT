import random
from difflib import SequenceMatcher

import torch
import numpy as np
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from scheduling_edict import EDICTScheduler

# Have diffusers with hardcoded double-casting instead of float
from my_diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                          LMSDiscreteScheduler, PNDMScheduler,
                          UNet2DConditionModel)
from my_diffusers.schedulers.scheduling_utils import SchedulerOutput

# StableDiffusion P2P implementation originally from https://github.com/bloc97/CrossAttentionControl




# Build our CLIP model
model_path_clip = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
clip = CLIPTextModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)


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
    target_prompt,
    guidance_scale=3,
    init_image_strength=0.8,
    steps=50,
    mix_weight=0.93,
    leapfrog_steps=True
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
    """
    # Resize/center crop to 512x512 (Can do higher res. if desired)
    orig_im = (
        load_im_into_format_from_path(im_path) if isinstance(im_path, str) else im_path
    )  # trust OK

    gen = coupled_stablediffusion(
        base_prompt=base_prompt,
        target_prompt=target_prompt,
        guidance_scale=guidance_scale,
        init_image=orig_im,
        init_image_strength=init_image_strength,
        steps=steps,
        mix_weight=mix_weight,
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

def encode_prompt(text_prompt):
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
        text_prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    embedding_conditional = clip(
        tokens_conditional.input_ids.to(device)
    ).last_hidden_state

    text_emb = torch.cat([embedding_unconditional, embedding_conditional])

    return text_emb



def run_noising_loop(latents, text_emb, timesteps, guidance_scale, leapfrog_steps, unet, scheduler):
    base, model_input = latents[1], latents[0]

    do_mixing_now = True
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

        # 1. Do mixing
        if do_mixing_now:
            base, model_input = scheduler.reverse_mixing_layer(base, model_input)

            # 2. Do swap after calling reverse_mixing_layer and before computing eq. (15.2)
            if leapfrog_steps:
                base, model_input = model_input, base
        
        # 3. Compute Unet
        latent_model_input = torch.cat([model_input] * 2)

        noise_pred = unet(
            latent_model_input, t, encoder_hidden_states=text_emb
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # 4. Do noise step
        base, model_input = scheduler.reverse_step(sample=base, model_input=model_input, model_output=noise_pred, timestep=t)
        do_mixing_now ^= True

    if not leapfrog_steps:
        base, model_input = model_input, base

    return base, model_input

@torch.no_grad()
def coupled_stablediffusion(
    base_prompt="",
    target_prompt="",
    guidance_scale=7.0,
    init_image=None,
    init_image_strength=1.0, # = s
    steps=50,
    mix_weight=0.93,
    leapfrog_steps=True,
    seed=1,
    width=512,
    height=512,
):
    # If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None:
        seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)

    def image_to_latent(im):
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
    
    # Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 8
    height = height - height % 8

    # Preprocess image if it exists (img2img)
    if isinstance(init_image, list):
            if isinstance(init_image[0], torch.Tensor):
                init_latent = [t.clone() for t in init_image]
            else:
                init_latent = [image_to_latent(im) for im in init_image]
    else:
        init_latent = image_to_latent(init_image)


    t_limit = steps - int(steps * init_image_strength)
        
    latent = init_latent
    if isinstance(latent, list):  # initializing from pair of images
        latent_pair = latent
    else:  # initializing from noise
        latent_pair = [latent.clone(), latent.clone()]


    # Set inference timesteps to scheduler
    scheduler = EDICTScheduler(mix_weight=mix_weight)
    scheduler.set_timesteps(steps)

    # CLIP Text Embeddings
    base_embeds = encode_prompt(base_prompt)
    target_embeds = encode_prompt(target_prompt)

    bwd_timesteps = scheduler.timesteps[t_limit:].repeat_interleave(2)  # 780, 780, ...
    fwd_timesteps = bwd_timesteps.flip(0)  # 0, 0, ....


    # Do noising loop
    base, model_input = run_noising_loop(latent_pair, base_embeds, fwd_timesteps, guidance_scale, leapfrog_steps, unet, scheduler)
    
    do_mixing_now = True
    # Do denoising loop (right here)
    for i, t in tqdm(enumerate(bwd_timesteps), total=len(bwd_timesteps)):
        

        # 1. Compute Unet
        latent_model_input = torch.cat([model_input] * 2)

        noise_pred = unet(
            latent_model_input, t, encoder_hidden_states=target_embeds
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 2. Do denoise step
        base, model_input = scheduler.forward_step(sample=base, model_input=model_input, model_output=noise_pred, timestep=t)
        do_mixing_now ^= True

        # 3. Do mixing
        if do_mixing_now:

            base, model_input = scheduler.forward_mixing_layer(base, model_input)
            # 4. Do swap after calling forward_mixing_layer and before computing (14.1) 
            # for the next iteration (that is, at the end of the current iteration).
            if leapfrog_steps:
                base, model_input = model_input, base

    latent_pair = [base, model_input]

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
