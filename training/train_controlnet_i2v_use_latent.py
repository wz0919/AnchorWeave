# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append('.')
sys.path.append('..')
import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from einops import rearrange
from decord import VideoReader
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
    # clear_objs_and_retain_memory,
)
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from controlnet_datasets_camera_pcd_mask import RealEstate10KPCDRenderLatentCapEmbDataset
from controlnet_pipeline import ControlnetCogVideoXImageToVideoPCDPipeline
from cogvideo_transformer import CustomCogVideoXTransformer3DModel
from cogvideo_controlnet_pcd import CogVideoXControlnetPCD
from anchor_processor import (
    apply_tube_mask,
    apply_frame_mask,
    apply_anchor_sampling,
    apply_first_anchor_frame_mask,
)


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--video_root_dir",
        type=str,
        default=None,
        required=True,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        required=False,
        help=("A path to csv."),
    )
    parser.add_argument(
        "--hflip_p",
        type=float,
        default=0.5,
        required=False,
        help="Video horizontal flip probability.",
    )
    parser.add_argument(
        "--use_camera_condition",
        action="store_true",
    )
    parser.add_argument(
        "--controlnet_transformer_num_layers",
        type=int,
        default=2,
        required=False,
        help=("Count of controlnet blocks."),
    )
    parser.add_argument(
        "--downscale_coef",
        type=int,
        default=8,
        required=False,
        help=("Downscale coef as encoder decreases resolutio before apply transformer."),
    )
    parser.add_argument(
        "--controlnet_input_channels",
        type=int,
        default=3,
        required=False,
        help=("Controlnet encoder input channels."),
    )
    parser.add_argument(
        "--controlnet_weights",
        type=float,
        default=1.0,
        required=False,
        help=("Controlnet blocks weight."),
    )
    parser.add_argument(
        "--init_from_transformer",
        action="store_true",
        help="Whether or not load start controlnet parameters from transformer model.",
    )
    parser.add_argument(
        "--pretrained_controlnet_path",
        type=str,
        default=None,
        required=False,
        help=("Path to controlnet .pt checkpoint."),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # Validation
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=(
            "Num steps for denoising on validation stage."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_video",
        type=str,
        default=None,
        help="Paths to video for falidation.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-controlnet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--enable_time_sampling",
        action="store_true",
        default=False,
        help="Whether or not to use time_sampling_dict.",
    )
    parser.add_argument(
        "--time_sampling_type",
        type=str,
        default="truncated_normal",
        choices=["truncated_normal", "truncated_uniform"]
    )
    parser.add_argument(
        "--time_sampling_mean",
        type=float,
        default=0.9,
        help="Shifted and truncated noise sampling",
    )
    parser.add_argument(
        "--time_sampling_std",
        type=float,
        default=0.03,
        help="Shifted and truncated noise sampling",
    )
    parser.add_argument(
        "--controlnet_guidance_end",
        type=float,
        default=0.2,
        help="Shifted and truncated noise sampling",
    )
    parser.add_argument(
        "--controlnet_guidance_start",
        type=float,
        default=0.0,
        help="Shifted and truncated noise sampling",
    )
    parser.add_argument(
        "--controlnet_transformer_num_attn_heads",
        type=int,
        default=None,
        required=False,
        help=("Count of attention heads in controlnet blocks."),
    )
    parser.add_argument(
        "--controlnet_transformer_attention_head_dim",
        type=int,
        default=None,
        required=False,
        help=("Attention dim in controlnet blocks."),
    )
    parser.add_argument(
        "--controlnet_transformer_out_proj_dim_factor",
        type=int,
        default=None,
        required=False,
        help=("Attention dim for custom controlnet blocks."),
    )
    parser.add_argument(
        "--controlnet_transformer_out_proj_dim_zero_init",
        action="store_true",
        default=False,
        help=("Init project zero."),
    )
    parser.add_argument(
        "--motion_sub_loss",
        action="store_true",
        default=False,
        help="Whether to use motion sub loss (frame difference loss).",
    )
    parser.add_argument(
        "--motion_sub_loss_ratio",
        type=float,
        default=0.2,
        help="Ratio of motion sub loss in total loss.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.995,
        help="EMA decay rate for controlnet when pretrained_controlnet_path is provided.",
    )
    parser.add_argument(
        "--ema_start_step",
        type=int,
        default=1500,
        help="Start EMA updates after this many training steps. Default 1500.",
    )

    return parser.parse_args()


def read_video(video_path, start_index=0, frames_count=49, stride=1):
    video_reader = VideoReader(video_path)
    end_index = min(start_index + frames_count * stride, len(video_reader)) - 1
    batch_index = np.linspace(start_index, end_index, frames_count, dtype=int)
    numpy_video = video_reader.get_batch(batch_index).asnumpy()
    return numpy_video
    

def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    for i, video in enumerate(videos):
        prompt = (
            pipeline_args["prompt"][:25]
            .replace(" ", "_")
            .replace(" ", "_")
            .replace("'", "_")
            .replace('"', "_")
            .replace("/", "_")
        )
        filename = os.path.join(args.output_dir, f"{epoch}_video_{i}_{prompt}.mp4")
        export_to_video(video, filename, fps=8)

    clear_objs_and_retain_memory([pipe])

    return videos


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

import numpy as np
import torch
import torch.nn.functional as F

def avgpool_mask_tensor(mask_tensor, output_h=30, output_w=45):
    bs, f, c, h, w = mask_tensor.shape
    assert c == 1, "Channel must be 1"
    assert f % 4 == 0, "Frame number must be divisible by 4 (e.g., 48)"
    assert h % output_h == 0 and w % output_w == 0, "Height and width must be divisible by 30 and 45"

    # # Binarize mask: values > 0.5 become 1, values <= 0.5 become 0
    # mask_tensor = (mask_tensor > 0.5).int()

    # Spatial average pooling
    x = mask_tensor.float()  # (bs, f, 1, h, w)
    x = x.view(bs * f, 1, h, w)
    x_pooled = F.avg_pool2d(x, kernel_size=(h // output_h, w // output_w))  # (bs * f, 1, 30, 45)
    x_pooled = x_pooled.view(bs, f, 1, output_h, output_w)

    # Temporal pooling
    latent_frames = f // 4
    x_pooled = x_pooled.view(bs, latent_frames, 4, 1, output_h, output_w)
    pooled_avg = torch.mean(x_pooled, dim=2)  # (bs, 12, 1, 30, 45)

    # Threshold
    pooled_mask = (pooled_avg > 0.5).int()

    # Add zero frame for each sample
    zero_frame = torch.zeros_like(pooled_mask[:, 0:1])  # (bs, 1, 1, 30, 45)
    pooled_mask = torch.cat([zero_frame, pooled_mask], dim=1)  # (bs, 13, 1, 30, 45)

    return 1 - pooled_mask  # invert

def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        tensorboard_log_dir = os.path.join(args.output_dir, "tensorboard_logs")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
                
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # # Prepare models and scheduler
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    # )

    # text_encoder = T5EncoderModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    # )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    controlnet_kwargs = {}
    num_attention_heads_orig = 48 if "5b" in args.pretrained_model_name_or_path.lower() else 30
    if args.controlnet_transformer_num_attn_heads is not None:
        controlnet_kwargs["num_attention_heads"] = args.controlnet_transformer_num_attn_heads
    else:
        controlnet_kwargs["num_attention_heads"] = num_attention_heads_orig
    if args.controlnet_transformer_attention_head_dim is not None:
        controlnet_kwargs["attention_head_dim"] = args.controlnet_transformer_attention_head_dim
    if args.controlnet_transformer_out_proj_dim_factor is not None:
        controlnet_kwargs["out_proj_dim"] = num_attention_heads_orig * args.controlnet_transformer_out_proj_dim_factor
    controlnet_kwargs["out_proj_dim_zero_init"] = args.controlnet_transformer_out_proj_dim_zero_init
    controlnet = CogVideoXControlnetPCD(
        num_layers=args.controlnet_transformer_num_layers,
        downscale_coef=args.downscale_coef,
        in_channels=args.controlnet_input_channels,
        use_camera_condition=args.use_camera_condition,
        **controlnet_kwargs,   
    )

    if args.init_from_transformer:
        controlnet_state_dict = {}
        for name, params in transformer.state_dict().items():
            if 'patch_embed.proj.weight' in name:
                continue
            controlnet_state_dict[name] = params
        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f'[ Weights from transformer was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')

    controlnet_ema = None
    use_ema = False
    if args.pretrained_controlnet_path:
        ckpt = torch.load(args.pretrained_controlnet_path, map_location='cpu', weights_only=False)
        controlnet_state_dict = {}
        for name, params in ckpt['state_dict'].items():
            controlnet_state_dict[name] = params
        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f'[ Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')
        
        # Create EMA model
        use_ema = True
        controlnet_ema = type(controlnet)(**controlnet.config)
        
        # Load EMA state dict if available, otherwise copy from controlnet
        if 'ema_state_dict' in ckpt:
            m_ema, u_ema = controlnet_ema.load_state_dict(ckpt['ema_state_dict'], strict=False)
            print(f'[ EMA model loaded from checkpoint ] [M: {len(m_ema)} | U: {len(u_ema)}]')
        else:
            controlnet_ema.load_state_dict(controlnet.state_dict())
            print(f'[ EMA model initialized from controlnet (no EMA in checkpoint) ]')
        
        controlnet_ema.requires_grad_(False)
        controlnet_ema.eval()
        print(f'[ EMA model created with decay rate: {args.ema_decay} ]')
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional adapter controlnet layers
    # text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.requires_grad_(True)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    
    # Move EMA model to the same device and dtype if it exists
    if use_ema and controlnet_ema is not None:
        controlnet_ema.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params([controlnet], dtype=torch.float32)

    trainable_parameters = list(filter(lambda p: p.requires_grad, controlnet.parameters()))

    # Optimization parameters
    trainable_parameters_with_lr = {"params": trainable_parameters, "lr": args.learning_rate}
    params_to_optimize = [trainable_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    train_dataset = RealEstate10KPCDRenderLatentCapEmbDataset(
        video_root_dir=args.video_root_dir
    )
        
    def encode_video(video):
        video = video.to(accelerator.device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
        return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)
    
    def collate_fn(examples):
        source_latents = [example["source_latent"] for example in examples]
        anchor_latents = [example["anchor_latents"] for example in examples]
        anchor_masks = [example["anchor_masks"] for example in examples]
        first_frames = [example["first_frame"] for example in examples]
        gt_cameras = [example["gt_and_first_frame_camera"] for example in examples]
        context_cameras = [example["context_cameras"] for example in examples]
        caption_embs = [example["caption_emb"] for example in examples]
        
        caption_embs = torch.concat(caption_embs)

        source_latents = torch.stack(source_latents)
        source_latents = source_latents.to(memory_format=torch.contiguous_format).float()

        anchor_latents = [torch.stack(latents) for latents in anchor_latents] 
        anchor_latents = torch.stack(anchor_latents) # [B, 4 (# anchors), F, C, H, W]
        anchor_latents = anchor_latents.to(memory_format=torch.contiguous_format).float()
        
        first_frames = torch.stack(first_frames).to(memory_format=torch.contiguous_format).float()  # [B, F, C, H, W]
        
        anchor_masks = torch.stack(anchor_masks) #[B, 4 (# anchors), F, H, W]
        anchor_masks = anchor_masks.to(memory_format=torch.contiguous_format).float()
        
        B, N = anchor_masks.shape[:2]

        anchor_masks = rearrange(anchor_masks, 'b n f h w -> (b n) f 1 h w')
        # found average pool works better than max pool
        anchor_masks = avgpool_mask_tensor(1-anchor_masks[:,1:], output_h=30, output_w=45)
        anchor_masks = rearrange(anchor_masks, '(b n) f 1 h w -> b n f 1 h w', b=B, n=N) 
        
        gt_cameras = torch.stack(gt_cameras)
        gt_cameras = gt_cameras.to(memory_format=torch.contiguous_format).float()
        
        context_cameras = [torch.stack([first_frame_camera] + context_camera) for first_frame_camera, context_camera in zip(gt_cameras, context_cameras)]
        context_cameras = torch.stack(context_cameras)
        context_cameras = context_cameras.to(memory_format=torch.contiguous_format).float()
        
        return {
            "source_latents": source_latents,
            "anchor_latents": anchor_latents,
            "first_frames": first_frames,
            "caption_embs": caption_embs,
            "anchor_masks": anchor_masks,
            "gt_cameras": gt_cameras,
            "context_cameras": context_cameras
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-controlnet"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [controlnet]

            with accelerator.accumulate(models_to_accumulate):
                model_input = batch['source_latents'].to(dtype=weight_dtype)  # [B, F, C, H, W]
                anchor_masks = batch["anchor_masks"].to(dtype=weight_dtype)  # [B, 4 (# anchors), F, 1, H, W]
                first_frames = batch["first_frames"].to(dtype=weight_dtype)  # [B, F, C, H, W]
                anchor_latents = batch["anchor_latents"].to(dtype=weight_dtype)  # [B, 4 (# anchors), F, C, H, W]
                gt_cameras = batch["gt_cameras"].to(dtype=weight_dtype)
                context_cameras = batch["context_cameras"].to(dtype=weight_dtype)
                prompt_embeds = batch["caption_embs"].to(weight_dtype)
                
                # Sample noise that will be added to the latents
                noise = torch.randn_like(model_input)
                batch_size, num_frames, num_channels, height, width = model_input.shape

                # Sample a random timestep for each image
                if args.enable_time_sampling:
                    if args.time_sampling_type == "truncated_normal":
                        time_sampling_dict = {
                            'mean': args.time_sampling_mean,
                            'std': args.time_sampling_std,
                            'a': 1 - args.controlnet_guidance_end,
                            'b': 1 - args.controlnet_guidance_start,
                        }
                        timesteps = torch.nn.init.trunc_normal_(
                            torch.empty(batch_size, device=model_input.device), **time_sampling_dict
                            ) * scheduler.config.num_train_timesteps
                    elif args.time_sampling_type == "truncated_uniform":
                        timesteps = torch.randint(
                            int((1- args.controlnet_guidance_end) * scheduler.config.num_train_timesteps),
                            int((1 - args.controlnet_guidance_start) * scheduler.config.num_train_timesteps),
                            (batch_size,), device=model_input.device
                        )
                else:    
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                    )
                timesteps = timesteps.long()
        
                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.height,
                        width=args.width,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

                images = batch["first_frames"][:,0].unsqueeze(2)
                # Add noise to images
                image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=accelerator.device)
                image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
                noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
                with torch.no_grad():
                    image_latent_dist = vae.encode(noisy_images.to(dtype=vae.dtype)).latent_dist
                    image_latents = image_latent_dist.sample() * vae.config.scaling_factor
                    image_latents = image_latents.permute(0, 2, 1, 3, 4)

                # Padding image_latents to the same frame number as latent
                padding_shape = (model_input.shape[0], model_input.shape[1] - 1, *model_input.shape[2:])
                latent_padding = image_latents.new_zeros(padding_shape)
                image_latents = torch.cat([image_latents, latent_padding], dim=1)

                # Concatenate latent and image_latents in the channel dimension
                latent_img_noisy = torch.cat([noisy_model_input, image_latents], dim=2)

                anchor_states = batch["anchor_latents"].to(dtype=weight_dtype)
                
                
                # conotext cameras: [B, # anchors, F, 3, 4]
                # controlnet input states: [B, # anchors, F, C, H, W]
                # anchor masks: [B, # anchors, F, 1, H, W]
                
                # Apply anchor processor with 0.2 probability for each method, 0.2 for no-op
                processor_choice = torch.rand(1).item()
                
                # if processor_choice < 0.2:
                #     # Apply tube mask
                #     controlnet_input_states, anchor_masks = apply_tube_mask(
                #         anchor_states.clone(),
                #         anchor_masks.clone(),
                #         mask_ratio=0.5,
                #         random_seed=None,
                #     )
                #     # context_cameras unchanged
                if processor_choice < 0.25:
                    # Apply frame mask
                    controlnet_input_states, anchor_masks = apply_frame_mask(
                        anchor_states.clone(),
                        anchor_masks.clone(),
                        mask_ratio=0.5,
                        random_seed=None,
                    )
                    # context_cameras unchanged
                elif processor_choice < 0.5:
                    # Apply anchor sampling
                    context_cameras, controlnet_input_states, anchor_masks = apply_anchor_sampling(
                        context_cameras.clone(),
                        anchor_states.clone(),
                        anchor_masks.clone(),
                        num_anchors=None,  # Random between 1 and original
                        random_seed=None,
                    )
                elif processor_choice < 0.75:
                    # Apply first anchor + frame mask
                    context_cameras, controlnet_input_states, anchor_masks = apply_first_anchor_frame_mask(
                        context_cameras.clone(),
                        anchor_states.clone(),
                        anchor_masks.clone(),
                        min_mask_ratio=0.25,
                        random_seed=None,
                    )
                else:
                    # No-op: keep original (0.2 probability)
                    controlnet_input_states = anchor_states
                    # anchor_masks and context_cameras remain unchanged
                
                controlnet_output = controlnet(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    image_rotary_emb=image_rotary_emb,
                    controlnet_states=controlnet_input_states,
                    gt_cameras=gt_cameras,
                    context_cameras=context_cameras,
                    anchor_masks=anchor_masks,
                    timestep=timesteps,
                    return_dict=False,
                )
                # controlnet_output is now (controlnet_hidden_states, camera_states_list)
                controlnet_states, camera_states_list = controlnet_output
                if isinstance(controlnet_states, (tuple, list)):
                    controlnet_states = [x.to(dtype=weight_dtype) for x in controlnet_states]
                else:
                    controlnet_states = controlnet_states.to(dtype=weight_dtype)
                if isinstance(camera_states_list, (tuple, list)):
                    camera_states_list = [x.to(dtype=weight_dtype) if x is not None else None for x in camera_states_list]
                elif camera_states_list is not None:
                    camera_states_list = camera_states_list.to(dtype=weight_dtype)
                model_output = transformer(
                    hidden_states=latent_img_noisy,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    controlnet_states=controlnet_states,
                    controlnet_weights=args.controlnet_weights,
                    camera_states_list=camera_states_list,
                    camera_weights=getattr(args, 'camera_weights', 1.0),
                    return_dict=False,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = model_input

                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                loss = loss.mean()
                
                # Motion sub loss: frame difference loss
                if args.motion_sub_loss:
                    # gt_sub_noise: ground truth frame differences (from target)
                    gt_sub_noise = target[:, 1:, :].float() - target[:, :-1, :].float()
                    # pred_sub_noise: predicted frame differences (from model_pred)
                    pred_sub_noise = model_pred[:, 1:, :].float() - model_pred[:, :-1, :].float()
                    sub_loss = F.mse_loss(pred_sub_noise, gt_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()
                
                if use_ema and controlnet_ema is not None and accelerator.sync_gradients and (global_step + 1) >= args.ema_start_step:
                    with torch.no_grad():
                        controlnet_unwrapped = unwrap_model(controlnet)
                        if (global_step + 1) == args.ema_start_step:
                            # First time enabling EMA: sync from controlnet (was not updated before)
                            controlnet_ema.load_state_dict(controlnet_unwrapped.state_dict())
                            if accelerator.is_main_process:
                                logger.info(f"EMA enabled at step {args.ema_start_step}, synced from controlnet")
                        else:
                            for ema_param, param in zip(controlnet_ema.parameters(), controlnet_unwrapped.parameters()):
                                ema_param.data.mul_(args.ema_decay).add_(param.data, alpha=1.0 - args.ema_decay)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                        checkpoint_dict = {'state_dict': unwrap_model(controlnet).state_dict()}
                        if use_ema and controlnet_ema is not None:
                            checkpoint_dict['ema_state_dict'] = controlnet_ema.state_dict()
                        torch.save(checkpoint_dict, save_path)
                        logger.info(f"Saved state to {save_path}")
                        
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if accelerator.is_main_process:
                writer.add_scalar("train/loss", logs["loss"], global_step)
                writer.add_scalar("train/learning_rate", logs["lr"], global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if args.validation_prompt is not None and (step + 1) % args.validation_steps == 0:
                    # Create pipeline
                    pipe = ControlnetCogVideoXPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=unwrap_model(transformer),
                        text_encoder=unwrap_model(text_encoder),
                        vae=unwrap_model(vae),
                        controlnet=unwrap_model(controlnet),
                        scheduler=scheduler,
                        torch_dtype=weight_dtype,
                    )
    
                    validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                    validation_videos = args.validation_video.split(args.validation_prompt_separator)
                    for validation_prompt, validation_video in zip(validation_prompts, validation_videos):
                        numpy_frames = read_video(validation_video, frames_count=args.max_num_frames)
                        controlnet_frames = np.stack([train_dataset.controlnet_processor(x) for x in numpy_frames])
                        pipeline_args = {
                            "prompt": validation_prompt,
                            "controlnet_frames": controlnet_frames,
                            "guidance_scale": args.guidance_scale,
                            "use_dynamic_cfg": args.use_dynamic_cfg,
                            "height": args.height,
                            "width": args.width,
                            "num_frames": args.max_num_frames,
                            "num_inference_steps": args.num_inference_steps,
                            "controlnet_weights": args.controlnet_weights,
                        }
    
                        validation_outputs = log_validation(
                            pipe=pipe,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            epoch=epoch,
                        )
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)