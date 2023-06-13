from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from typing import Any, Callable, Dict, List, Optional, Union
import inspect
import numpy as np
import PIL.Image
import torch
from dataclasses import dataclass
from diffusers.utils import deprecate, logging, BaseOutput
import cv2
from pdb import set_trace as st
from einops import rearrange, repeat


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    # panorama_height /= 8
    # panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []

    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views

def conbine_condition(conds):
    """
    conds: f x c x h x w
    """
    mean_cond = torch.mean(conds, dim=0)
    conds[0, :, :, :] = mean_cond
    return conds
    # return mean_cond * 0.5 + conds * 0.5


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]



class oursControlPipelineZeroFlicks(StableDiffusionControlNetPipeline):
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, controlnet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
        
    def DDPM_forward(self, x0, t0, tMax, generator, device, shape, text_embeddings):
        rand_device = "cpu" if device.type == "mps" else device

        if x0 is None:
            return torch.randn(shape, generator=generator, device=rand_device, dtype=text_embeddings.dtype).to(device)
        else:
            b, c, f, h, w = x0.shape
            # Method 1 
            eps = torch.randn([b, c, h, w], dtype=text_embeddings.dtype, generator=generator, device=rand_device)
            eps = repeat(eps, "b c h w -> b c f h w", f=f)
            # Method 2
            # eps = torch.randn(x0.shape, dtype=text_embeddings.dtype, generator=generator, device=rand_device)

            # breakpoint()
            alpha_vec = torch.prod(self.scheduler.alphas[t0:tMax])
            print(f"alpha_vec = {alpha_vec}")

            xt = torch.sqrt(alpha_vec) * x0 + \
                torch.sqrt(1-alpha_vec) * eps
            return xt
    
    def DDIM_backward(self, num_inference_steps, timesteps, skip_t, t0, t1, do_classifier_free_guidance, null_embs, text_embeddings, latents_local,
                      latents_dtype, guidance_scale, guidance_stop_step, callback, callback_steps, extra_step_kwargs, num_warmup_steps):
        entered = False

        f = latents_local.shape[2]

        latents_local = rearrange(latents_local, "b c f w h -> (b f) c w h")

        latents = latents_local.detach().clone()
        x_t0_1 = None
        x_t1_1 = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t > skip_t:
                    continue
                else:
                    if not entered:
                        print(
                            f"Continue DDIM with i = {i}, t = {t}, latent = {latents.shape}, device = {latents.device}, type = {latents.dtype}")
                        entered = True

                latents = latents.detach()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    if null_embs is not None:
                        text_embeddings[0] = null_embs[i][0]
                    te = torch.cat([repeat(text_embeddings[0, :, :], "c k -> f c k", f=f),
                                   repeat(text_embeddings[1, :, :], "c k -> f c k", f=f)])
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=te).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(
                        2)
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                if i >= guidance_stop_step * len(timesteps):
                    alpha = 0
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # latents = latents - alpha * grads / (torch.norm(grads) + 1e-10)
                # call the callback, if provided
                
                # if i < len(timesteps)-1 and timesteps[i+1] == t0:
                if i < len(timesteps)-1 and timesteps[i] == t0:
                    x_t0_1 = latents.detach().clone()
                    print(f"latent t0 found at i = {i}, t = {t}")
                # elif i < len(timesteps)-1 and timesteps[i+1] == t1:
                elif i < len(timesteps)-1 and timesteps[i] == t1:
                    x_t1_1 = latents.detach().clone()
                    print(f"latent t1 found at i={i}, t = {t}")

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        latents = rearrange(latents, "(b f) c w h -> b c f w h", f=f)

        res = {"x0": latents.detach().clone()}
        if x_t0_1 is not None:
            x_t0_1 = rearrange(x_t0_1, "(b f) c w h -> b c f  w h", f=f)
            res["x_t0_1"] = x_t0_1.detach().clone()
        if x_t1_1 is not None:
            x_t1_1 = rearrange(x_t1_1, "(b f) c w h -> b c f  w h", f=f)
            res["x_t1_1"] = x_t1_1.detach().clone()
        return res
    
    def warp_latents_with_flow(self, 
                               latents,
                               flow,):
        from utils import warp_latents_independently
        latents = rearrange(latents, "b c f w h -> (b f) c w h")
        out_latents = [latents]
        for idx in range(flow.shape[0]):
            warped = warp_latents_independently(out_latents[-1], flow[idx])
            # warped = rearrange(warped, "(b f) c w h -> b c f w h", f=latents.shape[0])
            # warped = warped[:, :, None, :, :]
            out_latents.append(warped)
            # st()
        out = torch.stack(out_latents, dim=2) 
        return out
    
    def get_flow_latents(self, 
                         latents, 
                         flow,
                         prompt_embeds,
                         do_classifier_free_guidance,
                         num_inference_steps,
                         timesteps,
                         guidance_scale,
                         callback, 
                         callback_steps,
                         extra_step_kwargs, 
                         num_warmup_steps,
                         generator,
                         device,
                         ):
        # shape of latents: (7, 4, 64, 64)
        f_latents = latents.shape[0]
        f_flow = flow.shape[0]
        
        assert f_latents - 2 == f_flow, "Flow and latents should have the same number of frames"
        f = f_latents - 1
        
        xT = latents[:1] # should be only the first latent 
        # st()
        xT = xT[:, :, None, :, :] # fake (b c f w h)
        shape = xT.shape
        
        null_embs = None
        text_embeddings = prompt_embeds
        dtype = prompt_embeds.dtype
        t0 = timesteps[3]
        t1 = timesteps[0]
        
        guidance_stop_step = 0.5

        ddim_res = self.DDIM_backward(num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=1000, t0=t0, t1=t1, do_classifier_free_guidance=do_classifier_free_guidance,
                                null_embs=null_embs, text_embeddings=text_embeddings, latents_local=xT, latents_dtype=dtype, guidance_scale=guidance_scale, guidance_stop_step=guidance_stop_step,
                                callback=callback, callback_steps=callback_steps, extra_step_kwargs=extra_step_kwargs, num_warmup_steps=num_warmup_steps)


        # x0 = ddim_res["x0"].detach() # only for visualization
        
        x_t0_1 = ddim_res["x_t0_1"].detach()
        x_t1_1 = ddim_res["x_t1_1"].detach()


        x_t0_k = self.warp_latents_with_flow(x_t0_1, flow)
        
        assert x_t0_k.shape == (x_t0_1.repeat(1, 1, f, 1, 1).shape)

        # x_t1_k = self.DDPM_forward(
        #     x0=x_t0_k, t0=t0, tMax=t1, device=device, shape=shape, text_embeddings=text_embeddings, generator=generator)
        
        x_t1 = torch.cat([x_t1_1, x_t0_k], dim=2).clone().detach()
        # st()
        x_t1 = rearrange(x_t1, 'b c f h w -> (b f) c h w')
        return x_t1
  
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        flow: Optional[List[torch.FloatTensor]] = None,
    ):
        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare image
        if isinstance(self.controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(self.controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        if flow is not None:
            latents = self.get_flow_latents(latents=latents,
                                            flow=flow,
                                            prompt_embeds=prompt_embeds,
                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                            num_inference_steps=num_inference_steps,
                                            timesteps=timesteps,
                                            guidance_scale=guidance_scale,
                                            callback=callback,
                                            callback_steps=callback_steps,
                                            extra_step_kwargs=extra_step_kwargs,
                                            num_warmup_steps=num_warmup_steps,
                                            generator=generator,
                                            device=device,
                                            )
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    controlnet_latent_model_input = latents
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    controlnet_latent_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                # 拆分chunk
                f, c, h, w = controlnet_latent_model_input.shape
                chunk_size = 8
                chunk_ids = np.arange(0, f, chunk_size)
                list_down_block_res_samples = []
                list_mid_block_res_sample = []

                for i in range(len(chunk_ids)):
                    ch_start = chunk_ids[i]
                    ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                    frame_ids = list(range(ch_start, ch_end))

                    latent_input = controlnet_latent_model_input[frame_ids]
                    prompt_input = controlnet_prompt_embeds[frame_ids]
                    image_input = image[frame_ids]

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_input,
                        t,
                        encoder_hidden_states=prompt_input,
                        controlnet_cond=image_input,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    for j in range(len(down_block_res_samples)):
                        if len(list_down_block_res_samples) <= j:
                            list_down_block_res_samples.append([])
                        list_down_block_res_samples[j].append(down_block_res_samples[j])
                    list_mid_block_res_sample.append(mid_block_res_sample)

                # 合并chunk
                down_block_res_samples = [torch.cat(d, dim=0) for d in list_down_block_res_samples]
                mid_block_res_sample = torch.cat(list_mid_block_res_sample, dim=0)

                # merge the condition
                down_block_res_samples = [conbine_condition(d) for d in down_block_res_samples]
                mid_block_res_sample = conbine_condition(mid_block_res_sample)

                if guess_mode and do_classifier_free_guidance:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # 拆分chunk
                f, c, h, w = controlnet_latent_model_input.shape
                f = f // 2 
                chunk_size = 8
                chunk_ids = np.arange(0, f, chunk_size - 1)
                # list_noise = []
                noise_pred = torch.zeros_like(latent_model_input)

                for i in range(len(chunk_ids)):
                    ch_start = chunk_ids[i]
                    ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                    frame_ids = [0] + list(range(ch_start, ch_end))
                    frame_ids = frame_ids + [ids + f for ids in frame_ids]

                    latent_input = latent_model_input[frame_ids]
                    prompt_input = prompt_embeds[frame_ids]
                    down_block_input = [down[frame_ids] for down in down_block_res_samples]
                    mid_block_input = mid_block_res_sample[frame_ids]
                    # image_input = image[frame_ids]


                    # predict the noise residual
                    noise_pred_ = self.unet(
                        latent_input,
                        t,
                        encoder_hidden_states=prompt_input,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_input,
                        mid_block_additional_residual=mid_block_input,
                    ).sample

                    # apply the noise pred to noise_pred with frame_ids
                    noise_pred[frame_ids] = noise_pred_

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
