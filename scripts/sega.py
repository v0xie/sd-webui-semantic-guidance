import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import scipy.stats as stats

from modules import script_callbacks, prompt_parser
from modules.script_callbacks import CFGDenoiserParams
from modules.prompt_parser import reconstruct_multicond_batch
from modules.processing import StableDiffusionProcessing
#from modules.shared import sd_model, opts
from modules.sd_samplers_cfg_denoiser import pad_cond
from modules import shared

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
                self.concept_name = ''
                self.v = {} # velocity
                self.warmup_period: int = 10 # [0, 20]
                self.edit_guidance_scale: float = 1 # [0., 1.]
                self.tail_percentage_threshold: float = 0.05 # [0., 1.] if abs value of difference between uncodition and concept-conditioned is less than this, then zero out the concept-conditioned values less than this
                self.momentum_scale: float = 0.3 # [0., 1.]
                self.momentum_beta: float = 0.6 # [0., 1.) # larger bm is less volatile changes in momentum
                self.strength = 1.0

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
                        with gr.Row():
                                prompt = gr.Textbox(lines=2, label="Prompt", elem_id = 'sega_prompt', info="Prompt goes here'", elem_classes=["prompt"])
                        with gr.Row():
                                neg_prompt = gr.Textbox(lines=2, label="Negative Prompt", elem_id = 'sega_neg_prompt', info="Negative Prompt goes here'", elem_classes=["prompt"])
                        with gr.Row():
                                warmup = gr.Slider(value = 10, minimum = 0, maximum = 30, step = 1, label="Warmup Period", elem_id = 'sega_warmup', info="How many steps to wait before applying semantic guidance, default 10")
                                edit_guidance_scale = gr.Slider(value = 1.0, minimum = 0.0, maximum = 20.0, step = 0.01, label="Edit Guidance Scale", elem_id = 'sega_edit_guidance_scale', info="Scale of edit guidance, default 1.0")
                                tail_percentage_threshold = gr.Slider(value = 0.05, minimum = 0.0, maximum = 1.0, step = 0.01, label="Tail Percentage Threshold", elem_id = 'sega_tail_percentage_threshold', info="The percentage of latents to modify, default 0.05")
                                momentum_scale = gr.Slider(value = 0.3, minimum = 0.0, maximum = 1.0, step = 0.01, label="Momentum Scale", elem_id = 'sega_momentum_scale', info="Scale of momentum, default 0.3")
                                momentum_beta = gr.Slider(value = 0.6, minimum = 0.0, maximum = 0.999, step = 0.01, label="Momentum Beta", elem_id = 'sega_momentum_beta', info="Beta for momentum, default 0.6")
                active.do_not_save_to_config = True
                prompt.do_not_save_to_config = True
                neg_prompt.do_not_save_to_config = True
                warmup.do_not_save_to_config = True
                edit_guidance_scale.do_not_save_to_config = True
                tail_percentage_threshold.do_not_save_to_config = True
                momentum_scale.do_not_save_to_config = True
                momentum_beta.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, 'SEGA Active'),
                        (prompt, 'SEGA Prompt'),
                        (neg_prompt, 'SEGA Negative Prompt'),
                        (warmup, 'SEGA Warmup Period'),
                        (edit_guidance_scale, 'SEGA Edit Guidance Scale'),
                        (tail_percentage_threshold, 'SEGA Tail Percentage Threshold'),
                        (momentum_scale, 'SEGA Momentum Scale'),
                        (momentum_beta, 'SEGA Momentum Beta'),
                ]
                self.paste_field_names = [
                        'sega_active',
                        'sega_prompt',
                        'sega_neg_prompt',
                        'sega_warmup',
                        'sega_edit_guidance_scale',
                        'sega_tail_percentage_threshold',
                        'sega_momentum_scale',
                        'sega_momentum_beta'
                ]
                return [active, prompt, neg_prompt, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta]

        def process_batch(self, p: StableDiffusionProcessing, active, prompt, neg_prompt, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta, *args, **kwargs):
                active = getattr(p, "sega_active", active)
                if active is False:
                        return
                # FIXME: must have some prompt
                prompt = getattr(p, "sega_prompt", prompt)
                #neg_prompt = getattr(p, "sega_neg_prompt", neg_prompt)
                #if prompt is None:
                #        return
                #if len(prompt) == 0:
                #        return
                steps = p.steps
                p.extra_generation_params = {
                        "SEGA Active": active,
                        "SEGA Prompt": prompt,
                        "SEGA Negative Prompt": neg_prompt,
                        "SEGA Warmup Period": warmup,
                        "SEGA Edit Guidance Scale": edit_guidance_scale,
                        "SEGA Tail Percentage Threshold": tail_percentage_threshold,
                        "SEGA Momentum Scale": momentum_scale,
                        "SEGA Momentum Beta": momentum_beta,
                }

                # separate concepts by comma
                concept_prompts = self.parse_concept_prompt(prompt)
                concept_prompts_neg = self.parse_concept_prompt(neg_prompt)
                # [[concept_1,  strength_1], ...]
                concept_prompts = [prompt_parser.parse_prompt_attention(concept)[0] for concept in concept_prompts]
                concept_prompts_neg = [prompt_parser.parse_prompt_attention(neg_concept)[0] for neg_concept in concept_prompts_neg]
                concept_prompts_neg = [[concept, -strength] for concept, strength in concept_prompts_neg]
                concept_prompts.extend(concept_prompts_neg)

                concept_conds = []
                for concept, strength in concept_prompts:
                        prompt_list = [concept] * p.batch_size
                        prompts = prompt_parser.SdConditioning(prompt_list, width=p.width, height=p.height)
                        c = p.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, steps, [self.cached_c], p.extra_network_data)
                        concept_conds.append([c, strength])

                self.create_hook(p, active, concept_conds, None, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta)

        def parse_concept_prompt(self, prompt:str) -> list[str]:
                """
                Separate prompt by comma into a list of concepts
                TODO: parse prompt into a list of concepts using A1111 functions
                >>> g = lambda prompt: self.parse_concept_prompt(prompt)
                >>> g("apples")
                ['apples']
                >>> g("apple, banana, carrot")
                ['apple', 'banana', 'carrot']
                """
                return [x.strip() for x in prompt.split(",")]

        def create_hook(self, p, active, concept_conds, concept_conds_neg, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta, *args, **kwargs):
                # Create a list of parameters for each concept
                concepts_sega_params = []
                for _, strength in concept_conds:
                        sega_params = SegaStateParams()
                        sega_params.warmup_period = warmup
                        sega_params.edit_guidance_scale = edit_guidance_scale
                        sega_params.tail_percentage_threshold = tail_percentage_threshold
                        sega_params.momentum_scale = momentum_scale
                        sega_params.momentum_beta = momentum_beta
                        sega_params.strength = strength
                        concepts_sega_params.append(sega_params)

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, concept_conds, concepts_sega_params)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)

        def postprocess_batch(self, p, active, neg_text, *args, **kwargs):
                active = getattr(p, "sega_active", active)
                if active is False:
                        return
                self.unhook_callbacks()

        def unhook_callbacks(self):
                logger.debug('Unhooked callbacks')
                script_callbacks.remove_current_script_callbacks()

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, concept_conds, sega_params: list[SegaStateParams]):
                # TODO: add option to opt out of batching for performance
                sampling_step = params.sampling_step
                text_cond = params.text_cond
                text_uncond = params.text_uncond

                # pad text_cond or text_uncond to match the length of the longest prompt
                # i would prefer to let sd_samplers_cfg_denoiser.py handle the padding, but
                # there isn't a callback that returns the padded conds
                if text_cond.shape[1] != text_uncond.shape[1]:
                        empty = shared.sd_model.cond_stage_model_empty_prompt
                        num_repeats = (text_cond.shape[1] - text_uncond.shape[1]) // empty.shape[1]

                        if num_repeats < 0:
                                text_cond = pad_cond(text_cond, -num_repeats, empty)
                        elif num_repeats > 0:
                                text_uncond = pad_cond(text_uncond, num_repeats, empty)

                batch_conds_list = []
                batch_tensor = {}

                for i, _ in enumerate(sega_params):
                        concept_cond, _ = concept_conds[i]
                        conds_list, tensor_dict = reconstruct_multicond_batch(concept_cond, sampling_step)
                        # initialize here because we don't know the shape/dtype of the tensor until we reconstruct it
                        for key, tensor in tensor_dict.items():
                                if tensor.shape[1] != text_uncond[key].shape[1]:
                                        empty = shared.sd_model.cond_stage_model_empty_prompt
                                        num_repeats = (tensor.shape[1] - text_uncond.shape[1]) // empty.shape[1]
                                        if num_repeats < 0:
                                                tensor = pad_cond(tensor, -num_repeats, empty)
                                tensor = tensor.unsqueeze(0)
                                if key not in batch_tensor.keys():
                                        batch_tensor[key] = tensor
                                else:
                                        batch_tensor[key] = torch.cat((batch_tensor[key], tensor), dim=0)
                        batch_conds_list.append(conds_list)
                self.sega_routine_batch(params, batch_conds_list, batch_tensor, sega_params)

        def sega_routine_batch(self, params: CFGDenoiserParams, batch_conds_list, batch_tensor, sega_params: list[SegaStateParams]):
                # FIXME: these parameters should be specific to each concept
                warmup_period = sega_params[0].warmup_period
                edit_guidance_scale = sega_params[0].edit_guidance_scale
                tail_percentage_threshold = sega_params[0].tail_percentage_threshold
                momentum_scale = sega_params[0].momentum_scale
                momentum_beta = sega_params[0].momentum_beta

                sampling_step = params.sampling_step
                text_cond = params.text_cond
                text_uncond = params.text_uncond

                # for dim = 4, new_shape will be (-1, 1, 1, 1), for dim=3, new_shape will be (-1, 1, 1), etc.
                make_tuple_dim = lambda dim: (-1,) + (1,) * (dim - 1)

                # Semantic Guidance
                edit_dir_dict = {}

                # batch_tensor: [num_concepts, batch_size, tokens(77, 154, etc.), 2048]
                # Calculate edit direction
                for key, concept_cond in batch_tensor.items():
                        #new_shape = (-1,) + (1,) * (concept_cond.dim() - 1)
                        new_shape = make_tuple_dim(concept_cond.dim())
                        strength = torch.Tensor([params.strength for params in sega_params]).to(dtype=concept_cond.dtype, device=concept_cond.device)
                        strength = strength.view(new_shape)

                        if key not in edit_dir_dict.keys():
                                edit_dir_dict[key] = torch.zeros_like(concept_cond, dtype=concept_cond.dtype, device=concept_cond.device)

                        # filter out values in-between tails
                        # FIXME: does this take into account image batch size?, i.e. dim 1
                        inside_dim = tuple(range(-concept_cond.dim() + 1, 0)) # for tensor of dim 4, returns (-3, -2, -1), for tensor of dim 3, returns (-2, -1)
                        cond_mean, cond_std = torch.mean(concept_cond, dim=inside_dim), torch.std(concept_cond, dim=inside_dim)

                        # broadcast element-wise subtraction
                        edit_dir = concept_cond - text_uncond[key]

                        # multiply by strength for positive / negative direction
                        edit_dir = torch.mul(strength, edit_dir)

                        # z-scores for tails
                        upper_z = stats.norm.ppf(1.0 - tail_percentage_threshold)

                        # numerical thresholds
                        # FIXME: does this take into account image batch size?, i.e. dim 1
                        upper_threshold = cond_mean + (upper_z * cond_std)

                        # reshape to be able to broadcast / use torch.where to filter out values for each concept
                        #new_shape = (-1,) + (1,) * (concept_cond.dim() - 1)
                        new_shape = make_tuple_dim(concept_cond.dim())
                        upper_threshold_reshaped = upper_threshold.view(new_shape)

                        # zero out values in-between tails
                        # elementwise multiplication between scale tensor and edit direction
                        zero_tensor = torch.zeros_like(concept_cond, dtype=concept_cond.dtype, device=concept_cond.device)
                        scale_tensor = torch.ones_like(concept_cond, dtype=concept_cond.dtype, device=concept_cond.device) * edit_guidance_scale
                        edit_dir_abs = edit_dir.abs()
                        scale_tensor = torch.where((edit_dir_abs > upper_threshold_reshaped), scale_tensor, zero_tensor)

                        # update edit direction with the edit dir for this concept
                        guidance_strength = 0.0 if sampling_step < warmup_period else 1.0 # FIXME: Use appropriate guidance strength
                        edit_dir = torch.mul(scale_tensor, edit_dir)
                        edit_dir_dict[key] = edit_dir_dict[key] + guidance_strength * edit_dir

                # TODO: batch this
                for i, sega_param in enumerate(sega_params):
                        for key, dir in edit_dir_dict.items():
                                # calculate momentum scale and velocity
                                if key not in sega_param.v.keys():
                                        slice_idx = 1 - dir.dim() # should be negative, for dim=4, slice_idx = -3
                                        sega_param.v[key] = torch.zeros(dir.shape[slice_idx:], dtype=dir.dtype, device=dir.device)

                                # add to text condition
                                v_t = sega_param.v[key]
                                dir[i] = dir[i] + torch.mul(momentum_scale, v_t)

                                # calculate v_t+1 and update state
                                v_t_1 = momentum_beta * ((1 - momentum_beta) * v_t) * dir[i]

                                # add to cond after warmup elapsed
                                if sampling_step >= warmup_period:
                                        text_cond[key] = text_cond[key] + dir[i]

                                # update velocity
                                sega_param.v[key] = v_t_1
