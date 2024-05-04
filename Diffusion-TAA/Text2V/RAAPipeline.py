
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from Text2V.models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from einops import rearrange, repeat

def load_primary_models(pretrained_model_path):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    path = r"/home/ubuntu/lileilei/sb_checkpoint"
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(path, subfolder='scheduler')
    return noise_scheduler, tokenizer, text_encoder, vae, unet, ddim_inv_scheduler

def vid2vid(
        pipeline, init_video, init_weight, prompt, negative_prompt, height, width, num_inference_steps,
        guidance_scale
):
    num_frames = init_video.shape[2]
    init_video = rearrange(init_video, "b c f h w -> (b f) c h w")
    latents = pipeline.vae.encode(init_video).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=num_frames)
    latents = pipeline.scheduler.add_noise(
        original_samples=latents * 0.18215,
        noise=torch.randn_like(latents),
        timesteps=(torch.ones(latents.shape[0]) * pipeline.scheduler.num_train_timesteps * (
                    1 - init_weight)).long(),
    )
    # if latents.shape[0] != len(prompt):
    #     latents = latents.repeat(len(prompt), 1, 1, 1, 1)

    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = pipeline._encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=latents.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    pipeline.scheduler.set_timesteps(num_inference_steps, device=latents.device)
    timesteps = pipeline.scheduler.timesteps
    timesteps = timesteps[round(init_weight * len(timesteps)):]

    with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
        for t in timesteps:
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # reshape latents
            bsz, channel, frames, width, height = latents.shape
            latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

            # reshape latents back
            latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

            progress_bar.update()

    video_tensor = pipeline.decode_latents(latents)

    return video_tensor
def cast_to_gpu_and_type(model_list, device, weight_dtype):
    for model in model_list:
        if model is not None: model.to(device, dtype=weight_dtype)

def RAA(pretrained_model_path,latents,prompt,device):
    noise_scheduler, tokenizer, text_encoder, vae, unet, ddim_inv_scheduler = load_primary_models(pretrained_model_path)
    models_to_cast = [text_encoder, vae,unet]
    cast_to_gpu_and_type(models_to_cast, device, weight_dtype=torch.float16)
    pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )
    pipeline.enable_vae_slicing()
    # ddim_inv_scheduler = ddim_inv_scheduler
    # ddim_inv_scheduler.set_timesteps(50)
    # diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    diffusion_scheduler = ddim_inv_scheduler
    pipeline.scheduler = diffusion_scheduler
    with torch.no_grad():
        videos = vid2vid(
            pipeline=pipeline,
            init_video=latents.to(device=latents.device, dtype=torch.half),
            init_weight=0.5,
            prompt=prompt,
            negative_prompt=None,
            height=224,
            width=224,
            num_inference_steps=50,
            guidance_scale=9.0,
        )
    del pipeline
    torch.cuda.empty_cache()
    return videos









if __name__=="__main__":
    latents=torch.randn(2,3,22,224,224)
    latents=latents / 127.5-1
    device=torch.device("cuda",3)
    latents=latents.to(device)
    prompt=['ego-car hits a car','wwww']
    path=r'/media/ubuntu/Seagate Expansion Drive/best_model/best_model1'
    out=RAA(path,latents,prompt,device)
    print(out.shape)
