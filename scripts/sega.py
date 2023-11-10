import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import numpy as np
from collections import OrderedDict
from typing import Union
import scipy.stats as stats

from modules import script_callbacks, rng, prompt_parser
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams, AfterCFGCallbackParams
from modules.prompt_parser import get_multicond_learned_conditioning, get_multicond_prompt_list, get_learned_conditioning_prompt_schedules, get_learned_conditioning, reconstruct_cond_batch, reconstruct_multicond_batch
from modules.processing import StableDiffusionProcessing
from modules.shared import sd_model

import torch

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""

An unofficial implementation of SEGA: Instructing Text-to-Image Models using Semantic Guidance for Automatic1111 WebUI

@misc{brack2023sega,
      title={SEGA: Instructing Text-to-Image Models using Semantic Guidance}, 
      author={Manuel Brack and Felix Friedrich and Dominik Hintersdorf and Lukas Struppek and Patrick Schramowski and Kristian Kersting},
      year={2023},
      eprint={2301.12247},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-semantic-guidance

"""

class SegaStateParams:
        def __init__(self):
                self.v = {} # velocity
                self.warmup_period: int = 5 # [0, 20]
                self.edit_guidance_scale: float = 1 # [0., 1.]
                self.tail_percentage_threshold: float = 0.25 # [0., 1.] if abs value of difference between uncodition and concept-conditioned is less than this, then zero out the concept-conditioned values less than this
                self.momentum_scale: float = 1.0 # [0., 1.]
                self.momentum_beta: float = 0.5 # [0., 1.) # larger bm is less volatile changes in momentum

class SegaExtensionScript(scripts.Script):
        def __init__(self):
                self.cached_c = [None, None]

        # Extension title in menu UI
        def title(self):
                return "Semantic Guidance"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion('Semantic Guidance', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='sega_active')
                        neg_text = gr.Textbox(lines=1, label="Prompt", elem_id = 'sega_neg_text', info="Prompt goes here'")
                        with gr.Row():
                                warmup = gr.Slider(value = 0.2, minimum = 0.0, maximum = 1.0, step = 0.01, label="Warmup Period", elem_id = 'sega_warmup', info="How many steps to wait before applying semantic guidance, default 5")
                                edit_guidance_scale = gr.Slider(value = 1.0, minimum = 0.0, maximum = 10.0, step = 0.01, label="Edit Guidance Scale", elem_id = 'sega_edit_guidance_scale', info="Scale of edit guidance, default 1.0")
                                tail_percentage_threshold = gr.Slider(value = 0.25, minimum = 0.0, maximum = 1.0, step = 0.01, label="Tail Percentage Threshold", elem_id = 'sega_tail_percentage_threshold', info="Threshold for tail percentage, default 0.25")
                                momentum_scale = gr.Slider(value = 1.0, minimum = 0.0, maximum = 1.0, step = 0.01, label="Momentum Scale", elem_id = 'sega_momentum_scale', info="Scale of momentum, default 1.0")
                                momentum_beta = gr.Slider(value = 0.5, minimum = 0.0, maximum = 0.999, step = 0.01, label="Momentum Beta", elem_id = 'sega_momentum_beta', info="Beta for momentum, default 0.5")
                active.do_not_save_to_config = True
                neg_text.do_not_save_to_config = True
                warmup.do_not_save_to_config = True
                edit_guidance_scale.do_not_save_to_config = True
                tail_percentage_threshold.do_not_save_to_config = True
                momentum_scale.do_not_save_to_config = True
                momentum_beta.do_not_save_to_config = True
                return [active, neg_text, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta]

        def process_batch(self, p: StableDiffusionProcessing, active, neg_text, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta, *args, **kwargs):
                active = getattr(p, "sega_active", active)
                if active is False:
                        return
                neg_text = getattr(p, "sega_neg_text", neg_text)
                steps = p.steps
                p.extra_generation_params = {
                        "SEGA Active": active,
                        "SEGA Negative Prompt": neg_text,
                        "SEGA Warmup Period": warmup,
                        "SEGA Edit Guidance Scale": edit_guidance_scale,
                        "SEGA Tail Percentage Threshold": tail_percentage_threshold,
                        "SEGA Momentum Scale": momentum_scale,
                        "SEGA Momentum Beta": momentum_beta,
                }

                #neg_text_ps = get_multicond_prompt_list([neg_text])
                prompts = prompt_parser.SdConditioning([neg_text], width=p.width, height=p.height)
                c = p.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, steps, [self.cached_c], p.extra_network_data)

                print('neg_text_ps', c)
                self.create_hook(p, active, c, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta)

        
        def create_hook(self, p, active, neg_text, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta, *args, **kwargs):
                # Use lambda to call the callback function with the parameters to avoid global variables
                sega_params = SegaStateParams()
                sega_params.warmup_period = warmup
                sega_params.edit_guidance_scale = edit_guidance_scale
                sega_params.tail_percentage_threshold = tail_percentage_threshold
                sega_params.momentum_scale = momentum_scale
                sega_params.momentum_beta = momentum_beta

                y = lambda params: self.on_cfg_denoiser_callback(params, neg_text, sega_params)
                #denoised_y = lambda params: self.on_cfg_denoised_callback(params, neg_text)
                #after_cfg_y = lambda params: self.on_cfg_after_cfg_callback(params, neg_text)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                #script_callbacks.on_cfg_denoised(denoised_y)
                #script_callbacks.on_cfg_after_cfg(after_cfg_y)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)

        def postprocess_batch(self, p, active, neg_text, *args, **kwargs):
                active = getattr(p, "sega_active", active)
                if active is False:
                        return
                self.unhook_callbacks()

        def unhook_callbacks(self):
                logger.debug('Unhooked callbacks')
                script_callbacks.remove_current_script_callbacks()

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, neg_text_ps, sega_params: SegaStateParams):
                total_sampling_steps = params.total_sampling_steps
                warmup_period = max(round(total_sampling_steps * sega_params.warmup_period), 0)
                edit_guidance_scale = sega_params.edit_guidance_scale
                tail_percentage_threshold = sega_params.tail_percentage_threshold
                momentum_scale = sega_params.momentum_scale
                momentum_beta = sega_params.momentum_beta

                x = params.x
                sampling_step = params.sampling_step
                text_cond = params.text_cond
                text_uncond = params.text_uncond

                conds_list, tensor = reconstruct_multicond_batch(neg_text_ps, sampling_step)

                edit_dir_dict = {}
                # Semantic Guidance
                # Calculate edit direction
                for key, cond in tensor.items():
                        if key not in edit_dir_dict.keys():
                                edit_dir_dict[key] = torch.zeros_like(text_cond[key], dtype=cond.dtype, device=cond.device)

                        # warmup period
                        #if sampling_step < warmup_period:
                        #        break

                        # filter out values in-between tails
                        cond_mean, cond_std = torch.mean(cond).item(), torch.std(cond).item()
                        edit_dir = cond - text_uncond[key]

                        # z-scores for tails
                        #lower_z = stats.norm.ppf(tail_percentage_threshold)
                        upper_z = stats.norm.ppf(1.0 - tail_percentage_threshold)

                        # numerical thresholds
                        #lower_threshold = cond_mean + (lower_z * cond_std)
                        upper_threshold = cond_mean + (upper_z * cond_std)
                        #lower_threshold *= edit_guidance_scale

                        # zero out values in-between tails
                        zero_tensor = torch.zeros_like(cond, dtype=cond.dtype, device=cond.device)
                        scale_tensor = torch.ones_like(cond, dtype=cond.dtype, device=cond.device) * edit_guidance_scale
                        edit_dir_abs = edit_dir.abs()
                        scale_tensor = torch.where((edit_dir_abs > upper_threshold), scale_tensor, zero_tensor)

                        # elementwise multiplication between scale tensor and edit direction
                        #jguidance_strength = 0.0 if sampling_step < warmup_period else conds_list[0][0][1] # FIXME: Use appropriate guidance strength
                        guidance_strength = 0.0 if sampling_step < warmup_period else 1.0 # FIXME: Use appropriate guidance strength
                        edit_dir = torch.mul(scale_tensor, edit_dir)
                        edit_dir_dict[key] = edit_dir_dict[key] + guidance_strength * edit_dir

                for key, dir in edit_dir_dict.items():
                        # multiply summed edit dir by text condition
                        #dir = dir * text_cond[key]

                        # calculate momentum scale and velocity
                        if key not in sega_params.v.keys():
                                sega_params.v[key] = torch.zeros_like(dir, dtype=dir.dtype, device=dir.device)

                        #text_cond[key] = dir + torch.mul(momentum_scale, sega_params.v[key])
                        # add to text condition
                        v_t = sega_params.v[key]
                        dir = dir + torch.mul(momentum_scale, v_t)

                        # calculate v_t+1 and update state
                        v_t_1 = momentum_beta * ((1 - momentum_beta) * v_t) * dir

                        if sampling_step >= warmup_period:
                                text_cond[key] = text_cond[key] + dir

                        sega_params.v[key] = v_t_1

        
        def calc_velocity(self, sampling_step, warmup_period, velocity_scale, v_t, v_0, last_eg):
                if sampling_step < warmup_period:
                        v_t = v_0
                # calculate semantic guidance term
                velocity = velocity_scale*v_t + (1-velocity_scale)*last_eg
                return velocity