import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import scipy.stats as stats

from modules import script_callbacks, prompt_parser, sd_hijack, sd_hijack_optimizations
from modules.hypernetworks import hypernetwork
#import modules.sd_hijack_optimizations
from ldm.util import default
from modules.script_callbacks import CFGDenoiserParams
from modules.prompt_parser import reconstruct_multicond_batch
from modules.processing import StableDiffusionProcessing
#from modules.shared import sd_model, opts
from modules.sd_samplers_cfg_denoiser import pad_cond
from modules import shared

import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import GaussianBlur

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
                self.dims = []

class SegaExtensionScript(scripts.Script):
        def __init__(self):
                self.cached_c = [None, None]
                self.handles = []

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
                                prompt = gr.Textbox(lines=2, label="Prompt", elem_id = 'sega_prompt', elem_classes=["prompt"])
                        with gr.Row():
                                neg_prompt = gr.Textbox(lines=2, label="Negative Prompt", elem_id = 'sega_neg_prompt', elem_classes=["prompt"])
                        with gr.Row():
                                warmup = gr.Slider(value = 10, minimum = 0, maximum = 30, step = 1, label="Window Size", elem_id = 'sega_warmup', info="How many steps to wait before applying semantic guidance, default 10")
                                edit_guidance_scale = gr.Slider(value = 1.0, minimum = 0.0, maximum = 20.0, step = 0.01, label="Correction Strength", elem_id = 'sega_edit_guidance_scale', info="Scale of edit guidance, default 1.0")
                                tail_percentage_threshold = gr.Slider(value = 0.05, minimum = 0.0, maximum = 1.0, step = 0.01, label="Alpha for CTNMS", elem_id = 'sega_tail_percentage_threshold', info="The percentage of latents to modify, default 0.05")
                                momentum_scale = gr.Slider(value = 0.3, minimum = 0.0, maximum = 1.0, step = 0.01, label="Correction Threshold", elem_id = 'sega_momentum_scale', info="Scale of momentum, default 0.3")
                                momentum_beta = gr.Slider(value = 0.6, minimum = 0.0, maximum = 0.999, step = 0.01, label="Correction Strength", elem_id = 'sega_momentum_beta', info="Beta for momentum, default 0.6")
                active.do_not_save_to_config = True
                prompt.do_not_save_to_config = True
                neg_prompt.do_not_save_to_config = True
                warmup.do_not_save_to_config = True
                edit_guidance_scale.do_not_save_to_config = True
                tail_percentage_threshold.do_not_save_to_config = True
                momentum_scale.do_not_save_to_config = True
                momentum_beta.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='SEGA Active' in d)),
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
                prompt = getattr(p, "sega_prompt", prompt)
                neg_prompt = getattr(p, "sega_neg_prompt", neg_prompt)
                warmup = getattr(p, "sega_warmup", warmup)
                edit_guidance_scale = getattr(p, "sega_edit_guidance_scale", edit_guidance_scale)
                tail_percentage_threshold = getattr(p, "sega_tail_percentage_threshold", tail_percentage_threshold)
                momentum_scale = getattr(p, "sega_momentum_scale", momentum_scale)
                momentum_beta = getattr(p, "sega_momentum_beta", momentum_beta)
                # FIXME: must have some prompt
                #if prompt is None:
                #        return
                #if len(prompt) == 0:
                #        return
                p.extra_generation_params.update({
                        "SEGA Active": active,
                        "SEGA Prompt": prompt,
                        "SEGA Negative Prompt": neg_prompt,
                        "SEGA Warmup Period": warmup,
                        "SEGA Edit Guidance Scale": edit_guidance_scale,
                        "SEGA Tail Percentage Threshold": tail_percentage_threshold,
                        "SEGA Momentum Scale": momentum_scale,
                        "SEGA Momentum Beta": momentum_beta,
                })

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
                        c = p.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, [self.cached_c], p.extra_network_data)
                        concept_conds.append([c, strength])

                self.create_hook(p, active, concept_conds, None, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta, p.width, p.height)

        def parse_concept_prompt(self, prompt:str) -> list[str]:
                """
                Separate prompt by comma into a list of concepts
                TODO: parse prompt into a list of concepts using A1111 functions
                >>> g = lambda prompt: self.parse_concept_prompt(prompt)
                >>> g("")
                []
                >>> g("apples")
                ['apples']
                >>> g("apple, banana, carrot")
                ['apple', 'banana', 'carrot']
                """
                if len(prompt) == 0:
                        return []
                return [x.strip() for x in prompt.split(",")]

        def create_hook(self, p, active, concept_conds, concept_conds_neg, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta, width, height, *args, **kwargs):
                # Create a list of parameters for each concept
                concepts_sega_params = []

                #for _, strength in concept_conds:
                sega_params = SegaStateParams()
                sega_params.warmup_period = warmup
                sega_params.edit_guidance_scale = edit_guidance_scale
                sega_params.tail_percentage_threshold = tail_percentage_threshold
                sega_params.momentum_scale = momentum_scale
                sega_params.momentum_beta = momentum_beta
                sega_params.strength = 1.0
                sega_params.dims = [width, height]
                concepts_sega_params.append(sega_params)

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, concept_conds, concepts_sega_params)
                un = lambda params: self.unhook_callbacks()

                # Hook callbacks
                if tail_percentage_threshold > 0:
                        self.ready_hijack_forward(sega_params, tail_percentage_threshold, width, height)

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

                num_handles = len(self.handles)
                for handle in self.handles:
                        # TODO: add unhook
                        handle.remove()
                        self.handles.remove(handle)
                print(f"Removed {num_handles} handles")
                script_callbacks.remove_current_script_callbacks()

        def correction_by_similarities(self, f, C, percentile, gamma, alpha):
                """
                Apply the Correction by Similarities algorithm on embeddings.

                Args:
                f (Tensor): The embedding tensor of shape (n, d).
                C (list): Indices of selected tokens.
                percentile (float): Percentile to use for score threshold.
                gamma (int): Window size for the windowing function.
                alpha (float): Correction strength.

                Returns:
                Tensor: The corrected embedding tensor.
                """
                if alpha == 0:
                        return f

                n, d = f.shape
                f_tilde = f.detach().clone()  # Copy the embedding tensor

                # Define a windowing function
                def psi(c, gamma, n, dtype, device):
                        window = torch.zeros(n, dtype=dtype, device=device)
                        start = max(0, c - gamma)
                        end = min(n, c + gamma + 1)
                        window[start:end] = 1
                        return window
                


                for token_idx, c in enumerate(C):
                        Sc = f[c] * f  # Element-wise multiplication
                        # product = greater positive value indicates more similarity
                        # filter out values under score threshold from 0 to max
                        Sc_flat_positive = Sc[Sc > 0]
                        k = 10
                        e= 2.718281
                        # 0.000001 < pct < 0.999999999
                        pct = min(0.999999999, max(0.000001, 1 - e**(-k * percentile))) 
                        tau = torch.quantile(Sc_flat_positive, pct)
                        Sc_tilde = Sc * (Sc > tau)  # Apply threshold and filter
                        Sc_tilde /= Sc_tilde.max()  # Normalize
                        window = psi(c, gamma, n, Sc_tilde.dtype, Sc_tilde.device).unsqueeze(1)  # Apply windowing function
                        Sc_tilde *= window
                        f_c_tilde = torch.sum(Sc_tilde * f, dim=0)  # Combine embeddings
                        f_tilde[c] = (1 - alpha) * f[c] + alpha * f_c_tilde  # Blend embeddings

                return f_tilde
        
        def ready_hijack_forward(self, sega_params, alpha, width, height):
                m = shared.sd_model
                nlm = m.network_layer_mapping
                cross_attn_modules = [m for m in nlm.values() if 'CrossAttention' in m.__class__.__name__]

                def cross_token_non_maximum_suppression(module, input, output):
                        batch_size, sequence_length, inner_dim = output.shape

                        max_dims = width*height
                        factor = math.isqrt(max_dims // sequence_length) # should be a square of 2
                        downscale_width = width // factor
                        downscale_height = height // factor

                        h = module.heads
                        head_dim = inner_dim // h
                        dtype = output.dtype
                        device = output.device

                        # Reshape the attention map to batch_size, height, width
                        attention_map = output.view(batch_size, downscale_height, downscale_width, inner_dim)
                        #attention_map = output.view(batch_size, sequence_length, h, head_dim)

                        # Select token indices (Assuming this is provided as sega_params or similar)
                        selected_tokens = torch.tensor(list(range(inner_dim)))  # Example: Replace with actual indices

                        # Extract and process the selected attention maps
                        gaussian_blur = GaussianBlur(kernel_size=3, sigma=1)
                        AC = attention_map[:, :, :, selected_tokens]  # Extracting relevant attention maps
                        AC = gaussian_blur(AC)  # Applying Gaussian smoothing

                        # Find the maximum contributing token for each pixel
                        M = torch.argmax(AC, dim=-1)

                        # Create one-hot vectors for suppression
                        t = attention_map.size(-1)
                        one_hot_M = F.one_hot(M, num_classes=t).to(dtype=dtype, device=device)

                        # Apply the suppression mask
                        #suppressed_attention_map = one_hot_M.unsqueeze(2) * attention_map
                        suppressed_attention_map = one_hot_M * attention_map

                        # Reshape back to original dimensions
                        suppressed_attention_map = suppressed_attention_map.view(batch_size, sequence_length, inner_dim)

                        out_tensor = (1-alpha) * output + alpha * suppressed_attention_map

                        return out_tensor

                for module in cross_attn_modules:
                        # TODO: add unhook
                        if len(module._forward_hooks) == 0:
                                handle = module.register_forward_hook(cross_token_non_maximum_suppression)
                                self.handles.append(handle)


        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, concept_conds, sega_params: list[SegaStateParams]):
                # TODO: add option to opt out of batching for performance
                sampling_step = params.sampling_step

                # SDXL
                if isinstance(params.text_cond, dict):
                        text_cond = params.text_cond['crossattn']
                        text_uncond = params.text_uncond['crossattn']
                # SD 1.5
                else:
                        text_cond = params.text_cond
                        text_uncond = params.text_uncond


                # correction by similarities
                text_cond_shape = text_cond.shape
                text_uncond_shape = text_cond.shape

                sp = sega_params[0]
                window_size = sp.warmup_period
                correction_strength = sp.momentum_beta
                score_threshold = sp.momentum_scale

                # [batch_size, tokens(77, 154, etc.), 2048]
                for batch_idx, batch in enumerate(text_cond):
                        #selected_token_idx = 0
                        #window_start_idx = max(0, selected_token_idx - window_size)
                        #window_end_idx = min(len(batch)-1, selected_token_idx + window_size)
                        #window = list(range(window_start_idx, window_end_idx))
                        window = list(range(0, len(batch)))

                        f_bar = self.correction_by_similarities(batch, window, score_threshold, window_size, correction_strength)

                        if isinstance(params.text_cond, dict):
                                params.text_cond['crossattn'][batch_idx] = f_bar
                        else:
                                params.text_cond[batch_idx] = f_bar

                        # # we want to select a token here, for nwo we do it on everything
                        # selected_token_idx = 0


                        # f = batch.detach().clone() # original embedding
                        # f_bar = batch.detach().clone() # corrected embedding, copy original batch

                        # # slice tokens within window
                        # window = batch[window_start_idx:window_end_idx]

                        # for token_index, embedding in enumerate(batch):
                        #         actual_token_idx = token_index + window_start_idx
                                
                        #         # line 4
                        #         s_c = embedding * f

                        #         # line 5 threshold and normalize
                        #         mask = s_c < score_threshold
                        #         s_c[mask] /= torch.max(s_c)

                        #         # line 6 windowing is done before this
                        #         s_c *= s_c

                        #         # line 7
                        #         f_c = f_bar[actual_token_idx]
                        #         n = len(batch)
                        #         sum_fc = n * (s_c * f) # sum of s_c * f from 1 to n
                        #         f_bar[actual_token_idx] = sum_fc[actual_token_idx]

                        #         # line 8
                        #         f_bar[actual_token_idx] = (1 - correction_strength) * f_c[actual_token_idx] + correction_strength * sum_fc[actual_token_idx]





                return
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

                # sd 1.5 support
                if isinstance(text_cond, torch.Tensor):
                        text_cond = {'crossattn': text_cond}
                if isinstance(text_uncond, torch.Tensor):
                        text_uncond = {'crossattn': text_uncond}

                for i, _ in enumerate(sega_params):
                        concept_cond, _ = concept_conds[i]
                        conds_list, tensor_dict = reconstruct_multicond_batch(concept_cond, sampling_step)

                        # sd 1.5 support
                        if isinstance(tensor_dict, torch.Tensor):
                                tensor_dict = {'crossattn': tensor_dict}

                        # initialize here because we don't know the shape/dtype of the tensor until we reconstruct it
                        for key, tensor in tensor_dict.items():
                                if tensor.shape[1] != text_uncond[key].shape[1]:
                                        empty = shared.sd_model.cond_stage_model_empty_prompt
                                        # sd 1.5
                                        if key == "crossattn":
                                                num_repeats = (tensor.shape[1] - text_uncond[key].shape[1]) // empty.shape[1]
                                        # sdxl
                                        else:
                                                num_repeats = (tensor.shape[1] - text_uncond.shape[1]) // empty.shape[1]
                                        if num_repeats < 0:
                                                tensor = pad_cond(tensor, -num_repeats, empty)
                                tensor = tensor.unsqueeze(0)
                                if key not in batch_tensor.keys():
                                        batch_tensor[key] = tensor
                                else:
                                        batch_tensor[key] = torch.cat((batch_tensor[key], tensor), dim=0)
                        batch_conds_list.append(conds_list)
                self.sega_routine_batch(params, batch_conds_list, batch_tensor, sega_params, text_cond, text_uncond)
        
        def make_tuple_dim(self, dim):
                # sd 1.5 support
                if isinstance(dim, torch.Tensor):
                        dim = dim.dim()
                return (-1,) + (1,) * (dim - 1)

        def sega_routine_batch(self, params: CFGDenoiserParams, batch_conds_list, batch_tensor, sega_params: list[SegaStateParams], text_cond, text_uncond):
                # FIXME: these parameters should be specific to each concept
                warmup_period = sega_params[0].warmup_period
                edit_guidance_scale = sega_params[0].edit_guidance_scale
                tail_percentage_threshold = sega_params[0].tail_percentage_threshold
                momentum_scale = sega_params[0].momentum_scale
                momentum_beta = sega_params[0].momentum_beta

                sampling_step = params.sampling_step

                # Semantic Guidance
                edit_dir_dict = {}

                # batch_tensor: [num_concepts, batch_size, tokens(77, 154, etc.), 2048]
                # Calculate edit direction
                for key, concept_cond in batch_tensor.items():
                        new_shape = self.make_tuple_dim(concept_cond)
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
                        new_shape = self.make_tuple_dim(concept_cond)
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
                                # for sd 1.5, we must add to the original params.text_cond because we reassigned text_cond
                                if sampling_step >= warmup_period:
                                        if isinstance(params.text_cond, dict):
                                                params.text_cond[key] = params.text_cond[key] + dir[i]
                                        else:
                                                params.text_cond = params.text_cond + dir[i]

                                # update velocity
                                sega_param.v[key] = v_t_1


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianSmoothing, self).__init__()
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        # This is the standard deviation of the normal distribution

        # Create a 1D Gaussian kernel
        kernel = 1 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(-torch.pow(torch.arange(kernel_size) - kernel_size // 2, 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        # Apply the kernel to each channel
        self.weight = nn.Parameter(kernel.view(1, 1, kernel_size).repeat(channels, 1, 1))

    def forward(self, x):
        # Apply 1D Gaussian smoothing
        return F.conv1d(x, self.weight, padding=self.kernel_size // 2)


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def sega_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
    return fun

def sega_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "sega_active"):
                setattr(p, "sega_active", True)
        setattr(p, field, x)

    return fun

def make_axis_options():
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
        extra_axis_options = {
                xyz_grid.AxisOption("[Semantic Guidance] Active", str, sega_apply_override('sega_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[Semantic Guidance] Prompt", str, sega_apply_field("sega_prompt")),
                xyz_grid.AxisOption("[Semantic Guidance] Negative Prompt", str, sega_apply_field("sega_neg_prompt")),
                xyz_grid.AxisOption("[Semantic Guidance] Guidance Scale", float, sega_apply_field("sega_edit_guidance_scale")),
                xyz_grid.AxisOption("[Semantic Guidance] Tail Percentage Threshold", float, sega_apply_field("sega_tail_percentage_threshold")),
                xyz_grid.AxisOption("[AAA] Window Size", int, sega_apply_field("sega_warmup")),
                xyz_grid.AxisOption("[AAA] Correction Threshold", float, sega_apply_field("sega_momentum_scale")),
                xyz_grid.AxisOption("[AAA] Correction Strength", float, sega_apply_field("sega_momentum_beta")),
        }
        if not any("[Semantic Guidance]" in x.label for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(extra_axis_options)

def callback_before_ui():
        try:
                make_axis_options()
        except:
                logger.exception("Semantic Guidance: Error while making axis options")

script_callbacks.on_before_ui(callback_before_ui)

def gaussian_smoothing(attention_maps, kernel_size=3, sigma=1):
    # Apply Gaussian smoothing
    channels = attention_maps.size(1)
    kernel = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = kernel_size // 2
    return F.conv2d(attention_maps, kernel, padding=padding, groups=channels)

# def cross_token_non_maximum_suppression(module, input, output):
        # batch_size, sequence_length, inner_dim = output.shape
        # h = module.heads
        # head_dim = inner_dim // h
        # dtype = output.dtype
        # device = output.device

        # # Reshape the attention map to separate heads
        # attention_map = output.view(batch_size, sequence_length, h, head_dim)

        # # Select token indices (Assuming this is provided as sega_params or similar)
        # selected_tokens = torch.tensor(list(range(head_dim)))  # Example: Replace with actual indices

        # # Extract and process the selected attention maps
        # gaussian_blur = GaussianBlur(kernel_size=3, sigma=1)
        # AC = attention_map[:, :, :, selected_tokens]  # Extracting relevant attention maps
        # AC = gaussian_blur(AC)  # Applying Gaussian smoothing

        # # Find the maximum contributing token for each pixel
        # M = torch.argmax(AC, dim=-1)

        # # Create one-hot vectors for suppression
        # t = attention_map.size(-1)
        # one_hot_M = F.one_hot(M, num_classes=t).to(dtype=dtype, device=device)

        # # Apply the suppression mask
        # #suppressed_attention_map = one_hot_M.unsqueeze(2) * attention_map
        # suppressed_attention_map = one_hot_M * attention_map

        # # Reshape back to original dimensions
        # suppressed_attention_map = suppressed_attention_map.view(batch_size, sequence_length, inner_dim)

        # alpha = 0.1
        # out_tensor = (1-alpha) * output + alpha * suppressed_attention_map

        # return out_tensor
