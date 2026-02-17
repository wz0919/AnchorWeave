import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.append('.')
sys.path.append('..')
import argparse
import os

import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    AutoencoderKLCogVideoX
)
from diffusers.utils import export_to_video, load_video

from controlnet_pipeline import ControlnetCogVideoXImageToVideoPCDPipeline
from cogvideo_transformer import CustomCogVideoXTransformer3DModel
from cogvideo_controlnet_pcd import CogVideoXControlnetPCD
from training.controlnet_datasets_camera_pcd_mask import RealEstate10KPCDRenderDataset
from torchvision.transforms.functional import to_pil_image

from inference.utils import stack_images_horizontally, stack_images_horizontally_multiple, stack_images_vertically, add_text_label
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def get_black_region_mask_tensor(video_tensor, threshold=2, kernel_size=15):
    """
    Generate cleaned binary masks for black regions in a video tensor.
    
    Args:
        video_tensor (torch.Tensor): shape (T, H, W, 3), RGB, uint8
        threshold (int): pixel intensity threshold to consider a pixel as black (default: 20)
        kernel_size (int): morphological kernel size to smooth masks (default: 7)
    
    Returns:
        torch.Tensor: binary mask tensor of shape (T, H, W), where 1 indicates black region
    """
    video_uint8 = ((video_tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)  # shape (T, H, W, C)
    video_np = video_uint8.numpy()

    T, H, W, _ = video_np.shape
    masks = np.empty((T, H, W), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for t in range(T):
        img = video_np[t]  # (H, W, 3), uint8
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks[t] = (mask_cleaned > 0).astype(np.uint8)
    return torch.from_numpy(masks)

def maxpool_mask_tensor(mask_tensor, output_h=30, output_w=45):
    """
    Apply spatial and temporal max pooling to a binary mask tensor.
    
    Args:
        mask_tensor (torch.Tensor): shape (bs, f, 1, h, w) or (f, 1, h, w), binary mask (0 or 1)
        output_h (int): output height after pooling
        output_w (int): output width after pooling
    
    Returns:
        torch.Tensor: shape (bs, 13, 1, output_h, output_w) or (13, 1, output_h, output_w), pooled binary mask
    """
    # Handle both (bs, f, 1, h, w) and (f, 1, h, w) formats
    if mask_tensor.dim() == 5:
        bs, f, c, h, w = mask_tensor.shape
        assert c == 1, "Channel must be 1"
        has_batch = True
    elif mask_tensor.dim() == 4:
        f, c, h, w = mask_tensor.shape
        assert c == 1, "Channel must be 1"
        has_batch = False
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, f, 1, h, w)
        bs = 1
    else:
        raise ValueError(f"Unsupported tensor dimension: {mask_tensor.dim()}, expected 4 or 5")
    
    assert f % 4 == 0, "Frame number must be divisible by 4 (e.g., 48)"
    assert h % output_h == 0 and w % output_w == 0, f"Height and width must be divisible by {output_h} and {output_w}"

    # # Binarize mask: all non-zero values become 1
    # mask_tensor = (mask_tensor != 0).int()

    # Spatial max pooling
    x = mask_tensor.float()  # (bs, f, 1, h, w)
    x = x.view(bs * f, 1, h, w)
    x_pooled = F.max_pool2d(x, kernel_size=(h // output_h, w // output_w))  # (bs * f, 1, output_h, output_w)
    x_pooled = x_pooled.view(bs, f, 1, output_h, output_w)

    # Temporal pooling
    latent_frames = f // 4
    x_pooled = x_pooled.view(bs, latent_frames, 4, 1, output_h, output_w)
    pooled_mask = torch.amax(x_pooled, dim=2)  # (bs, latent_frames, 1, output_h, output_w)

    # Add zero frame for each sample
    zero_frame = torch.zeros_like(pooled_mask[:, 0:1])  # (bs, 1, 1, output_h, output_w)
    pooled_mask = torch.cat([zero_frame, pooled_mask], dim=1)  # (bs, 13, 1, output_h, output_w)

    if not has_batch:
        pooled_mask = pooled_mask.squeeze(0)  # (13, 1, output_h, output_w)

    return 1 - pooled_mask.int()  # invert

def avgpool_mask_tensor(mask_tensor, output_h=30, output_w=45):
    """
    Apply spatial and temporal average pooling to a binary mask tensor,
    and threshold at 0.5 to retain only majority-active regions.
    
    Args:
        mask_tensor (torch.Tensor): shape (bs, f, 1, h, w) or (f, 1, h, w), binary mask (0 or 1)
        output_h (int): output height after pooling
        output_w (int): output width after pooling
    
    Returns:
        torch.Tensor: shape (bs, 13, 1, output_h, output_w) or (13, 1, output_h, output_w), pooled binary mask
    """
    # Handle both (bs, f, 1, h, w) and (f, 1, h, w) formats
    if mask_tensor.dim() == 5:
        bs, f, c, h, w = mask_tensor.shape
        assert c == 1, "Channel must be 1"
        has_batch = True
    elif mask_tensor.dim() == 4:
        f, c, h, w = mask_tensor.shape
        assert c == 1, "Channel must be 1"
        has_batch = False
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, f, 1, h, w)
        bs = 1
    else:
        raise ValueError(f"Unsupported tensor dimension: {mask_tensor.dim()}, expected 4 or 5")
    
    assert f % 4 == 0, "Frame number must be divisible by 4 (e.g., 48)"
    assert h % output_h == 0 and w % output_w == 0, f"Height and width must be divisible by {output_h} and {output_w}"

    # # Binarize mask: values > 0.5 become 1, values <= 0.5 become 0
    # mask_tensor = (mask_tensor > 0.5).int()

    # Spatial average pooling
    x = mask_tensor.float()  # (bs, f, 1, h, w)
    x = x.view(bs * f, 1, h, w)
    x_pooled = F.avg_pool2d(x, kernel_size=(h // output_h, w // output_w))  # (bs * f, 1, output_h, output_w)
    x_pooled = x_pooled.view(bs, f, 1, output_h, output_w)

    # Temporal pooling
    latent_frames = f // 4
    x_pooled = x_pooled.view(bs, latent_frames, 4, 1, output_h, output_w)
    pooled_avg = torch.mean(x_pooled, dim=2)  # (bs, latent_frames, 1, output_h, output_w)

    # Threshold
    pooled_mask = (pooled_avg > 0.5).int()

    # Add zero frame for each sample
    zero_frame = torch.zeros_like(pooled_mask[:, 0:1])  # (bs, 1, 1, output_h, output_w)
    pooled_mask = torch.cat([zero_frame, pooled_mask], dim=1)  # (bs, 13, 1, output_h, output_w)

    if not has_batch:
        pooled_mask = pooled_mask.squeeze(0)  # (13, 1, output_h, output_w)

    return 1 - pooled_mask  # invert

@torch.no_grad()
def generate_video(
    prompt,
    image,
    video_root_dir: str,
    base_model_path: str,
    use_camera_condition: bool,
    controlnet_model_path: str,
    split_type: str = "train",
    controlnet_weights: float = 1.0,
    controlnet_guidance_start: float = 0.0,
    controlnet_guidance_end: float = 1.0,
    use_dynamic_cfg: bool = True,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output/",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    num_frames: int = 49,
    height: int = 480,
    width: int = 720,
    start_camera_idx: int = 0,
    end_camera_idx: int = 1,
    controlnet_transformer_num_attn_heads: int = None,
    controlnet_transformer_attention_head_dim: int = None,
    controlnet_transformer_out_proj_dim_factor: int = None,
    controlnet_transformer_out_proj_dim_zero_init: bool = False,
    controlnet_transformer_num_layers: int = 8,
    downscale_coef: int = 8,
    controlnet_input_channels: int = 6,
    infer_with_mask: bool = False,
    pool_style: str = 'avg',
    pipe_cpu_offload: bool = False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - video_root_dir (str): The path to the camera dataset
    - annotation_json (str): Name of subset (train.json or test.json)
    - base_model_path (str): The path of the pre-trained model to be used.
    - controlnet_model_path (str): The path of the pre-trained conrolnet model to be used.
    - controlnet_weights (float): Strenght of controlnet
    - controlnet_guidance_start (float): The stage when the controlnet starts to be applied
    - controlnet_guidance_end (float): The stage when the controlnet end to be applied
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    os.makedirs(output_path, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(
        base_model_path, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder"
    )
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer"
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        base_model_path, subfolder="vae"
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )
    num_attention_heads_orig = 48 if "5b" in base_model_path.lower() else 30
    controlnet_kwargs = {}
    if controlnet_transformer_num_attn_heads is not None:
        controlnet_kwargs["num_attention_heads"] = args.controlnet_transformer_num_attn_heads
    else:
        controlnet_kwargs["num_attention_heads"] = num_attention_heads_orig
    if controlnet_transformer_attention_head_dim is not None:
        controlnet_kwargs["attention_head_dim"] = controlnet_transformer_attention_head_dim
    if controlnet_transformer_out_proj_dim_factor is not None:
        controlnet_kwargs["out_proj_dim"] = num_attention_heads_orig * controlnet_transformer_out_proj_dim_factor
    controlnet_kwargs["out_proj_dim_zero_init"] = controlnet_transformer_out_proj_dim_zero_init
    controlnet = CogVideoXControlnetPCD(
        num_layers=controlnet_transformer_num_layers,
        downscale_coef=downscale_coef,
        in_channels=controlnet_input_channels,
        use_camera_condition=use_camera_condition,
        **controlnet_kwargs,   
    )
    if controlnet_model_path:
        ckpt = torch.load(controlnet_model_path, map_location='cpu', weights_only=False)
        use_ema = 'ema_state_dict' in ckpt
        if use_ema:
            print('[ eval ema ]')
            controlnet_state_dict = {}
            for name, params in ckpt['ema_state_dict'].items():
                controlnet_state_dict[name] = params
        else:
            controlnet_state_dict = {}
            for name, params in ckpt['state_dict'].items():
                controlnet_state_dict[name] = params
        
        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        if use_ema:
            print(f'[ Weights from pretrained controlnet EMA was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')
        else:
            print(f'[ Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')
    
    pipe = ControlnetCogVideoXImageToVideoPCDPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        controlnet=controlnet,
        scheduler=scheduler,
    )
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # pipe.to("cuda")
    pipe = pipe.to(dtype=dtype)
    # pipe.enable_sequential_cpu_offload()
    # if pipe_cpu_offload:
    pipe.enable_model_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    eval_dataset = RealEstate10KPCDRenderDataset(
        video_root_dir=video_root_dir,
        image_size=(height, width), 
        sample_n_frames=num_frames,
        split_type=split_type,
    )
    
    None_prompt = True
    if prompt:
        None_prompt = False
    # print(eval_dataset.data_list)
    
    for camera_idx in range(start_camera_idx, end_camera_idx):
        data_dict = eval_dataset[camera_idx]
        reference_video = data_dict['source_video']
        anchor_videos_list = data_dict['anchor_videos']  # List of 4 videos, each [C, T, H, W]
        input_images = data_dict['first_frame']
        
        context_cameras_list = data_dict['context_cameras']  # List of 3 cameras, each [F, ...]
        gt_and_first_frame_camera = data_dict['gt_and_first_frame_camera']  # [F, ...]
        
        print(eval_dataset.dataset[camera_idx],seed)
        anchor_videos = torch.stack(anchor_videos_list)  # [4, C, T, H, W]
        context_cameras = torch.stack([gt_and_first_frame_camera] + context_cameras_list)  # [4, F, ...]
        context_cameras = context_cameras.unsqueeze(0)  # [1, 4, F, ...]
        gt_cameras = gt_and_first_frame_camera.unsqueeze(0)  # [1, F, ...]
        
        if None_prompt:
            output_path_file = os.path.join(output_path, f"{camera_idx:05d}_{seed}_out.mp4")
            prompt = data_dict['caption']
        else:
            output_path_file = os.path.join(output_path, f"{prompt[:10]}_{camera_idx:05d}_{seed}_out.mp4")

        if infer_with_mask:
            try:
                video_mask = data_dict['anchor_masks']
                video_mask = 1 - video_mask
            except:
                print('using derived mask')
                anchor_video_for_mask = anchor_videos[0].permute(1, 0, 2, 3)  # [T, C, H, W] for mask function
                video_mask = get_black_region_mask_tensor(anchor_video_for_mask)
            
            if pool_style == 'max':
                # video_mask[:,1:] is [4, T-1, H, W], need to add channel dimension: [4, T-1, 1, H, W]
                video_mask_input = video_mask[:,1:].unsqueeze(2).contiguous()  # [4, T-1, 1, H, W]
                anchor_masks = maxpool_mask_tensor(video_mask_input, output_h=30, output_w=45)  # [4, 13, 1, 30, 45]
                # anchor_masks = anchor_masks.flatten().unsqueeze(0).unsqueeze(-1).to('cuda')
            elif pool_style == 'avg':
                video_mask_input = video_mask[:,1:].unsqueeze(2).contiguous()  # [4, T-1, 1, H, W]
                anchor_masks = avgpool_mask_tensor(video_mask_input, output_h=30, output_w=45)  # [4, 13, 1, 30, 45]
                # anchor_masks = anchor_masks.flatten().unsqueeze(0).unsqueeze(-1).to('cuda')
        else:
            anchor_masks = None
        # if os.path.isfile(output_path_file):
        #     continue
        
        video_generate_all = pipe(
            image=input_images,
            anchor_videos=anchor_videos,
            anchor_masks=anchor_masks,
            context_cameras=context_cameras,
            gt_cameras=gt_cameras,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
            use_dynamic_cfg=use_dynamic_cfg,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            controlnet_weights=controlnet_weights,
            controlnet_guidance_start=controlnet_guidance_start,
            controlnet_guidance_end=controlnet_guidance_end,
        ).frames
        video_generate = video_generate_all[0]

        reference_frames = [to_pil_image(frame) for frame in ((reference_video.permute(1, 0, 2, 3)/2+0.5))]
        
        output_path_file_reference = output_path_file.replace("_out.mp4", "_reference.mp4")
        output_path_file_out_reference = output_path_file.replace(".mp4", "_reference.mp4")
        
        export_to_video(video_generate, output_path_file, fps=8)
        export_to_video(reference_frames, output_path_file_reference, fps=8)
        anchor_videos_frames_list = [[to_pil_image(frame) for frame in ((anchor_video.permute(1, 0, 2, 3)/2+0.5))] for anchor_video in anchor_videos_list[1:]]
        
        anchor_video_vis = anchor_videos[0].permute(1, 0, 2, 3)  # [T, C, H, W] for visualization
        anchor_video_first = [to_pil_image(frame) for frame in ((anchor_video_vis)/2+0.5)]
        
        reference_frames_labeled = [add_text_label(frame, "reference", position="top") for frame in reference_frames]
        video_generate_labeled = [add_text_label(frame, "generated", position="top") for frame in video_generate]
        anchor_video_first_labeled = [add_text_label(frame, "init frame anchor", position="top") for frame in anchor_video_first]
        
        first_row_frames = [
            stack_images_horizontally_multiple([
                reference_frames_labeled[i],
                video_generate_labeled[i],
                anchor_video_first_labeled[i]
            ], spacing=0)
            for i in range(len(video_generate))
        ]
        anchor_labels = ["context anchor 1", "context anchor 2", "context anchor 3"]
        anchor_videos_labeled = []
        for idx, anchor_frames in enumerate(anchor_videos_frames_list):
            label = anchor_labels[idx] if idx < len(anchor_labels) else f"anchor {idx+1}"
            anchor_videos_labeled.append([add_text_label(frame, label, position="top") for frame in anchor_frames])
        
        second_row_frames = [
            stack_images_horizontally_multiple([
                anchor_videos_labeled[0][i],
                anchor_videos_labeled[1][i],
                anchor_videos_labeled[2][i]
            ], spacing=0)
            for i in range(len(anchor_videos_labeled[0]))
        ]
        out_reference_frames = [
            stack_images_vertically(first_row, second_row)
            for first_row, second_row in zip(first_row_frames, second_row_frames)
        ]
        export_to_video(out_reference_frames, output_path_file_out_reference, fps=8)
        
        anchor_videos_frames_list = [anchor_video_first] + anchor_videos_frames_list
        all_anchors_frames = [
            stack_images_horizontally_multiple([
                anchor_videos_frames_list[0][i],
                anchor_videos_frames_list[1][i],
                anchor_videos_frames_list[2][i],
                anchor_videos_frames_list[3][i]
            ], spacing=0)
            for i in range(len(anchor_videos_frames_list[0]))
        ]
        output_path_file_anchors = output_path_file.replace(".mp4", "_anchors.mp4")
        export_to_video(all_anchors_frames, output_path_file_anchors, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, default=None, help="The description of the video to be generated")
    parser.add_argument("--image", type=str, default=None, help="The reference image of the video to be generated")
    parser.add_argument(
        "--video_root_dir",
        type=str,
        required=True,
        help="The path of the video for controlnet processing.",
    )
    parser.add_argument(
        "--base_model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_model_path", type=str, default="TheDenk/cogvideox-5b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument("--controlnet_weights", type=float, default=0.5, help="Strenght of controlnet")
    parser.add_argument("--use_camera_condition", action="store_true", default=False, help="Use zero conv")
    parser.add_argument("--split_type", type=str, default="train", choices=["train", "test"], help="Dataset split to use: 'train' or 'test'.")
    parser.add_argument("--infer_with_mask", action="store_true", default=False, help="add mask to controlnet")
    parser.add_argument("--pool_style", default='max', help="max pool or avg pool")
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0, help="The stage when the controlnet starts to be applied")
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.5, help="The stage when the controlnet end to be applied")
    parser.add_argument("--use_dynamic_cfg", type=bool, default=True, help="Use dynamic cfg")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--start_camera_idx", type=int, default=0)
    parser.add_argument("--end_camera_idx", type=int, default=1)
    parser.add_argument("--controlnet_transformer_num_attn_heads", type=int, default=None)
    parser.add_argument("--controlnet_transformer_attention_head_dim", type=int, default=None)
    parser.add_argument("--controlnet_transformer_out_proj_dim_factor", type=int, default=None)
    parser.add_argument("--controlnet_transformer_out_proj_dim_zero_init", action="store_true", default=False, help=("Init project zero."),
    )
    parser.add_argument("--downscale_coef", type=int, default=8)
    parser.add_argument("--vae_channels", type=int, default=16)
    parser.add_argument("--controlnet_input_channels", type=int, default=6)
    parser.add_argument("--controlnet_transformer_num_layers", type=int, default=8)
    parser.add_argument("--enable_model_cpu_offload", action="store_true", default=False, help="Enable model CPU offload")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        image=args.image,
        video_root_dir=args.video_root_dir,
        base_model_path=args.base_model_path,
        use_camera_condition=args.use_camera_condition,
        controlnet_model_path=args.controlnet_model_path,
        split_type=args.split_type,
        controlnet_weights=args.controlnet_weights,
        controlnet_guidance_start=args.controlnet_guidance_start,
        controlnet_guidance_end=args.controlnet_guidance_end,
        use_dynamic_cfg=args.use_dynamic_cfg,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        start_camera_idx=args.start_camera_idx,
        end_camera_idx=args.end_camera_idx,
        controlnet_transformer_num_attn_heads=args.controlnet_transformer_num_attn_heads,
        controlnet_transformer_attention_head_dim=args.controlnet_transformer_attention_head_dim,
        controlnet_transformer_out_proj_dim_factor=args.controlnet_transformer_out_proj_dim_factor,
        controlnet_transformer_num_layers=args.controlnet_transformer_num_layers,
        downscale_coef=args.downscale_coef,
        controlnet_input_channels=args.controlnet_input_channels,
        infer_with_mask=args.infer_with_mask,
        pool_style=args.pool_style,
        pipe_cpu_offload=args.enable_model_cpu_offload,
    )
