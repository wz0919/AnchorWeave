import os
import random
import json
import glob

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2

from decord import VideoReader
from packaging import version as pver
from safetensors.torch import load_file
from torch.utils.data.dataset import Dataset

def blur_anchor_video(anchor_np, blur_kernel_size=(25, 25)):
    """
    Apply Gaussian blur to anchor video numpy array.
    
    Args:
        anchor_np: numpy array of shape [T, H, W, C] with values in [0, 255] uint8
        blur_kernel_size: tuple of (kernel_height, kernel_width), must be odd numbers
    
    Returns:
        blurred_anchor_np: numpy array of same shape as input, with blur applied
    """
    T, H, W, C = anchor_np.shape
    
    blurred_frames = []
    for t in range(T):
        frame = anchor_np[t]  # [H, W, C], uint8
        # Apply Gaussian blur directly (cv2 handles RGB correctly)
        blurred_frame = cv2.GaussianBlur(frame, blur_kernel_size, 0)
        blurred_frames.append(blurred_frame)
    
    # Stack frames
    blurred_np = np.stack(blurred_frames, axis=0)  # [T, H, W, C]
    
    return blurred_np


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    """
    Compute Plucker embeddings from camera intrinsics and poses.
    
    Args:
        K: Camera intrinsics of shape (B, V, 4) where 4 = [fx, fy, cx, cy]
        c2w: Camera-to-world matrices of shape (B, V, 4, 4)
        H: Image height
        W: Image width
        device: Device to compute on
        flip_flag: Optional flip flags of shape (V,) for horizontal flipping
        
    Returns:
        plucker: Plucker embeddings of shape (B, V, H, W, 6)
    """
    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    return plucker


class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)

class RealEstate10KPCDRenderDataset(Dataset):
    def __init__(
            self,
            video_root_dir,
            sample_n_frames=49,
            image_size=[480, 720],
            shuffle_frames=False,
            hflip_p=0.0,
            split_type: str = "train",
    ):
        if hflip_p != 0.0:
            use_flip = True
        else:
            use_flip = False
        root_path = video_root_dir
        self.root_path = root_path
        self.sample_n_frames = sample_n_frames
        self.height, self.width = image_size
    
        split_list = os.path.join(root_path, f'{split_type}.txt')
        if os.path.exists(split_list):
            with open(split_list, 'r') as f:
                self.dataset = [line.strip() for line in f.read().splitlines() if line.strip()]
        else:
            self.dataset = [folder_name for folder_name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder_name))]
        
        self.length = len(self.dataset)
        self.use_flip = use_flip
        
        pixel_transforms = [transforms.Resize(image_size),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.pixel_transforms = pixel_transforms

    def compute_plucker_embedding(self, clip_name, indices):
        """
        Compute Plucker embedding from camera folder npz files.
        
        Args:
            clip_name: Name of the clip folder
            indices: Frame indices to sample (numpy array)
            
        Returns:
            plucker_embedding: torch.Tensor of shape (T, H, W, 6) where T = len(indices)
        """
        camera_dir = os.path.join(self.root_path, clip_name, 'camera')
        if not os.path.exists(camera_dir):
            raise FileNotFoundError(f"Camera directory not found: {camera_dir}")
        
        camera_files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
        if len(camera_files) == 0:
            raise FileNotFoundError(f"No camera files found in {camera_dir}")
        
        camera_poses = []
        camera_intrinsics_list = []
        
        for idx in indices:
            if idx >= len(camera_files):
                raise IndexError(f"Frame index {idx} out of range (max: {len(camera_files)-1})")
            camera_file = camera_files[idx]
            camera_data = np.load(camera_file)
            pose = camera_data['pose']  # (4, 4)
            intrinsics = camera_data['intrinsics']  # (3, 3)
            
            camera_poses.append(pose)
            camera_intrinsics_list.append(intrinsics)
        
        camera_poses = np.array(camera_poses)  # (T, 4, 4)
        camera_intrinsics_3x3 = np.array(camera_intrinsics_list)  # (T, 3, 3)
        
        # Extract [fx, fy, cx, cy] from camera intrinsics matrix
        fx = camera_intrinsics_3x3[:, 0, 0]  # shape: (T,)
        fy = camera_intrinsics_3x3[:, 1, 1]  # shape: (T,)
        cx = camera_intrinsics_3x3[:, 0, 2]  # shape: (T,)
        cy = camera_intrinsics_3x3[:, 1, 2]  # shape: (T,)
        
        # Scale intrinsics to target image size
        # Assuming original intrinsics are for 512x288 or similar, scale to target size
        camera_intrinsics = np.stack([
            fx / 512 * self.width,   # fx scaled to target width
            fy / 288 * self.height,  # fy scaled to target height
            cx / 512 * self.width,   # cx scaled to target width
            cy / 288 * self.height   # cy scaled to target height
        ], axis=1)  # shape: (T, 4)
        
        # Convert to relative poses (all poses relative to first frame)
        source_cam_c2w = camera_poses[0]  # First frame's c2w
        zero_t_first_frame = True  # Set to True to zero translation of first frame
        if zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        
        # Compute w2c for first frame
        source_cam_w2c = np.linalg.inv(source_cam_c2w)
        
        # Create target camera c2w (first frame at origin or with z offset)
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Compute transformation from absolute to relative
        abs2rel = target_cam_c2w @ source_cam_w2c
        
        # Convert all poses to relative poses
        relative_camera_poses = []
        relative_camera_poses.append(target_cam_c2w)  # First frame
        for abs_c2w in camera_poses[1:]:
            rel_c2w = abs2rel @ abs_c2w
            relative_camera_poses.append(rel_c2w)
        relative_camera_poses = np.array(relative_camera_poses, dtype=np.float32)  # (T, 4, 4)
        
        # Convert to torch tensors
        relative_camera_poses_torch = torch.from_numpy(relative_camera_poses).float()  # (T, 4, 4)
        camera_intrinsics_torch = torch.from_numpy(camera_intrinsics).float()  # (T, 4)
        
        # Add batch and view dimensions: (1, T, 4, 4) and (1, T, 4)
        camera_poses_batch = relative_camera_poses_torch.unsqueeze(0)  # (1, T, 4, 4)
        camera_intrinsics_batch = camera_intrinsics_torch.unsqueeze(0)  # (1, T, 4)
        
        # Compute Plucker embeddings
        device = 'cpu'  # Use CPU for computation
        plucker_embedding = ray_condition(
            camera_intrinsics_batch,
            camera_poses_batch,
            self.height,
            self.width,
            device=device,
            flip_flag=None
        )  # (1, T, H, W, 6)
        
        # Remove batch dimension: (T, H, W, 6)
        plucker_embedding = plucker_embedding[0]  # (T, H, W, 6)
        
        return plucker_embedding

    def get_batch(self, idx):
        """
        Returns (formatted for transforms):
            source_video: torch.Tensor, shape [T, C, H, W] - source video pixels (ready for transforms)
            anchor_videos: list of torch.Tensor, each shape [T, C, H, W] - anchor video pixels (ready for transforms)
            anchor_masks: torch.Tensor, shape [4, T, H, W] - anchor masks
            first_frame: torch.Tensor, shape [1, C, H, W] - first frame of source video (ready for transforms)
            gt_and_first_frame_camera: torch.Tensor - camera poses for GT and first frame
            context_cameras: list of torch.Tensor - camera poses for context frames
            caption: str - caption text
            clip_name: str - clip name
        """
        clip_name = self.dataset[idx]
        
        # caption path
        caption_path = os.path.join(self.root_path, clip_name, 'captions', 'caption.txt')
        
        # video paths
        source_video_path = os.path.join(self.root_path, clip_name, 'videos', 'GT.mp4')
        anchor_video_path = os.path.join(self.root_path, clip_name, 'videos', '1st_frame.mp4')
        context_video_paths = [os.path.join(self.root_path, clip_name, 'videos', f'retrieval_{i+1}.mp4') for i in range(3)]
        
        # masks paths
        anchor_mask_path = os.path.join(self.root_path, clip_name, 'masks', '1st_frame_mask.npz')
        context_masks_path = [os.path.join(self.root_path, clip_name, 'masks', f'retrieval_{i+1}_mask.npz') for i in range(3)]
        
        # camera paths
        first_frame_camera_path = os.path.join(self.root_path, clip_name, 'relative_cameras', f'1st_frame.npz')
        context_camera_paths = [os.path.join(self.root_path, clip_name, 'relative_cameras', f'retrieval_{i+1}.npz') for i in range(3)]
        
        # Load caption
        caption = open(caption_path, 'r').read().strip()
        
        # Load source video to get video length and compute indices
        source_video_reader = VideoReader(source_video_path)
        video_length = len(source_video_reader)
        indices = np.linspace(0, video_length - 1, self.sample_n_frames, dtype=int)
        
        # Source video: [T, H, W, C] -> [T, C, H, W] (ready for transforms)
        source_video_np = source_video_reader.get_batch(indices).asnumpy()  # [T, H, W, C]
        source_video = torch.from_numpy(source_video_np).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        source_video = source_video / 255.0
        
        # First frame: [1, C, H, W] (ready for transforms)
        first_frame = source_video[0:1]  # [1, C, H, W]
        
        # Load anchor videos
        anchor_video_reader = VideoReader(anchor_video_path)
        anchor_video_np = anchor_video_reader.get_batch(indices).asnumpy()  # [T, H, W, C]
        anchor_video = torch.from_numpy(anchor_video_np).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        anchor_video = anchor_video / 255.0
        
        context_videos = []
        for context_video_path in context_video_paths:
            context_video_reader = VideoReader(context_video_path)
            context_video_np = context_video_reader.get_batch(indices).asnumpy()  # [T, H, W, C]
            context_video = torch.from_numpy(context_video_np).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
            context_video = context_video / 255.0
            context_videos.append(context_video)
        
        anchor_videos = [anchor_video] + context_videos  # List of 4 videos, each [T, C, H, W]
        
        # Load masks
        anchor_mask_1st_frame = np.load(anchor_mask_path)['mask'].astype(np.float32)
        anchor_masks_retrieved_frames = [np.load(path)['mask'].astype(np.float32) for path in context_masks_path]
        anchor_masks = [anchor_mask_1st_frame] + anchor_masks_retrieved_frames
        anchor_masks = torch.from_numpy(np.stack(anchor_masks))  # [4, T, H, W]
        anchor_masks = anchor_masks[:, indices]  # Select frames according to indices
        
        # Load cameras
        gt_and_first_frame_camera = torch.from_numpy(np.load(first_frame_camera_path)['pose'])
        context_cameras = [torch.from_numpy(np.load(path)['pose']) for path in context_camera_paths]
        
        # Select frames according to indices
        gt_and_first_frame_camera = gt_and_first_frame_camera[indices]
        context_cameras = [camera[indices] for camera in context_cameras]
        
        return source_video, anchor_videos, anchor_masks, first_frame, gt_and_first_frame_camera, context_cameras, caption, clip_name

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        while True:
            try:
                source_video, anchor_videos, anchor_masks, first_frame, gt_and_first_frame_camera, context_cameras, caption, clip_name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)
        
        # Apply transforms to source video: [T, C, H, W] -> transform -> [C, T, H, W]
        T, C, H, W = source_video.shape
        for transform in self.pixel_transforms:
            source_video = transform(source_video)
        source_video = source_video.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        # Apply transforms to anchor videos: [T, C, H, W] -> transform -> [C, T, H, W]
        transformed_anchor_videos = []
        for anchor_video in anchor_videos:
            for transform in self.pixel_transforms:
                anchor_video = transform(anchor_video)
            transformed_anchor_videos.append(anchor_video.permute(1, 0, 2, 3))  # [C, T, H, W]
        
        # Apply transforms to first frame: [1, C, H, W] -> transform -> [C, 1, H, W]
        for transform in self.pixel_transforms:
            first_frame = transform(first_frame)
        first_frame = first_frame.squeeze(0)  # [C, H, W]
        
        # Apply resize to anchor masks: [4, T, H, W] -> [4*T, 1, H, W] -> resize -> [4, T, H', W']
        anchor_masks = anchor_masks.reshape(-1, H, W).unsqueeze(1)  # [4*T, 1, H, W]
        anchor_masks = self.pixel_transforms[0](anchor_masks)  # Resize only: [4*T, 1, H', W']
        # Get resized dimensions
        _, _, resized_h, resized_w = anchor_masks.shape
        anchor_masks = anchor_masks.squeeze(1).reshape(4, self.sample_n_frames, resized_h, resized_w)  # [4, T, H', W']
        
        data = {
            'source_video': source_video,  # [C, T, H, W]
            'anchor_videos': transformed_anchor_videos,  # List of 4 videos, each [C, T, H, W]
            'anchor_masks': anchor_masks,  # [4, T, H, W]
            'first_frame': first_frame,  # [C, H, W]
            'gt_and_first_frame_camera': gt_and_first_frame_camera,
            'context_cameras': context_cameras,
            'caption': caption,
            'clip_name': clip_name
        }
        return data
    
class RealEstate10KPCDRenderCapEmbDataset(RealEstate10KPCDRenderDataset):
    def __init__(
            self,
            video_root_dir,
            sample_n_frames=49,
            image_size=[480, 720],
            shuffle_frames=False,
            hflip_p=0.0,
    ):
        super().__init__(
            video_root_dir,
            sample_n_frames=sample_n_frames,
            image_size=image_size,
            shuffle_frames=shuffle_frames,
            hflip_p=hflip_p,
        )

    def get_batch(self, idx):
        clip_name, data_item_direction = self.data_list[idx]
        item_folder_path = os.path.join(self.root_path, clip_name)
        video_folder_path = os.path.join(item_folder_path, 'videos')
        caption_embed_folder_path = os.path.join(item_folder_path, 'caption_embs_cogvideox')
        caption_folder_path = os.path.join(item_folder_path, 'captions')
        
        video_path = os.path.join(video_folder_path, 'source_high_res.mp4')
        render_video_path = os.path.join(video_folder_path, 'render.mp4')
        mask_path = os.path.join(video_folder_path, 'mask.npz')
        video_reader = VideoReader(video_path)
        render_video_reader = VideoReader(render_video_path)

        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool)
        
        video_length = len(video_reader)
        indices = np.linspace(0, video_length - 1, self.sample_n_frames, dtype=int)
        
        if data_item_direction == 'forward':
            caption_embed_path = os.path.join(caption_embed_folder_path, 'caption.pt')
            caption_path = os.path.join(caption_folder_path, 'caption.txt')
            caption = open(caption_path, 'r').read().strip()
            video_caption_emb = torch.load(caption_embed_path, weights_only=True)
        elif data_item_direction == 'backward':
            caption_embed_path = os.path.join(caption_embed_folder_path, 'caption_reverse   .pt')
            caption_path = os.path.join(caption_folder_path, 'caption_reverse.txt')
            caption = open(caption_path, 'r').read().strip()
            caption_emb = torch.load(caption_embed_path, weights_only=True)
            indices = indices[::-1]
        else:
            raise ValueError(f"Invalid data item direction: {data_item_direction}")

        pixel_values = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        
        # Get anchor video as numpy array, blur it, then convert to tensor
        anchor_np = render_video_reader.get_batch(indices).asnumpy()  # [T, H, W, C], uint8
        anchor_np = blur_anchor_video(anchor_np, blur_kernel_size=(25, 25))  # [T, H, W, C], uint8
        anchor_pixels = torch.from_numpy(anchor_np).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        anchor_pixels = anchor_pixels / 255.
        
        try:
            masks = np.load(mask_path)['mask']*1.0
            masks = torch.from_numpy(masks).unsqueeze(1)
            masks = masks[indices]
        except:
            threshold = 0.1  # you can adjust this value
            masks = (anchor_pixels.sum(dim=1, keepdim=True) < threshold).float()
        
        return pixel_values, anchor_pixels, masks, video_caption_emb, flip_flag, clip_name
    
    def __getitem__(self, idx):
        while True:
            try:
                video, anchor_video, mask, video_caption_emb, flip_flag, clip_name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)
                
        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            video = self.pixel_transforms[2](video)
            anchor_video = self.pixel_transforms[0](anchor_video)
            anchor_video = self.pixel_transforms[1](anchor_video, flip_flag)
            anchor_video = self.pixel_transforms[2](anchor_video)
            mask = self.pixel_transforms[0](mask)
            mask = self.pixel_transforms[1](mask, flip_flag)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)
                anchor_video = transform(anchor_video)
            mask = self.pixel_transforms[0](mask)
        data = {
            'video': video, 
            'anchor_video': anchor_video,
            'caption_emb': video_caption_emb, 
            'mask': mask
        }
        return data

class RealEstate10KPCDRenderLatentCapEmbDataset(Dataset):
    def __init__(
            self,
            video_root_dir,
            split='train',
            sample_n_frames=49,
            image_size=[480, 720],
    ):
        root_path = video_root_dir
        self.root_path = root_path
        self.sample_n_frames = sample_n_frames
        self.height, self.width = image_size
        
        pixel_transforms = [transforms.Resize(image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.pixel_transforms = pixel_transforms
        
        # Load dataset list based on split
        split_list = os.path.join(root_path, f'{split}.txt')
        if os.path.exists(split_list):
            with open(split_list, 'r') as f:
                all_items = [line.strip() for line in f.read().splitlines() if line.strip()]
        else:
            raise ValueError(f"Split file {split_list} does not exist")

        # Split into dataset1 (without RealEstate) and dataset2 (only RealEstate)
        self.dataset = [item for item in all_items if 'RealEstate' not in item]
        self.dataset2 = [item for item in all_items if 'RealEstate' in item]
        
        print(f"Dataset1 (without RealEstate) length: {len(self.dataset)}")
        print(f"Dataset2 (only RealEstate) length: {len(self.dataset2)}")
        
        # Total length: dataset1.length / 3 * 4 = dataset1.length * 4/3
        self.length = len(self.dataset) * 4 // 3
        print(f"Total dataset length (3:1 ratio): {self.length}")

    def _idx_to_clip(self, idx):
        """
        Map global idx to (dataset_idx, clip_idx) where:
        - dataset_idx: 0 for dataset1, 1 for dataset2
        - clip_idx: index within the selected dataset
        Ratio: 3 dataset1 items for every 1 dataset2 item
        """
        group_idx = idx // 4
        pos_in_group = idx % 4
        
        if pos_in_group < 3:  # 0, 1, 2 -> dataset1
            dataset_idx = 0
            clip_idx = group_idx * 3 + pos_in_group
            clip_idx = clip_idx % len(self.dataset)  # wrap around if needed
        else:  # 3 -> dataset2
            dataset_idx = 1
            clip_idx = group_idx % len(self.dataset2)
        
        return dataset_idx, clip_idx

    def get_batch(self, idx):
        dataset_idx, clip_idx = self._idx_to_clip(idx)
        clip_name = self.dataset[clip_idx] if dataset_idx == 0 else self.dataset2[clip_idx]
        
        # caption embedding
        caption_emb_path = os.path.join(self.root_path, clip_name, 'caption_embs_cogvideox', 'caption.pt')
        
        # video path
        video_path = os.path.join(self.root_path, clip_name, 'videos', 'GT.mp4')
        # latents
        source_latent_path = os.path.join(self.root_path, clip_name, 'latents_cogvideox', 'GT.pt')
        anchor_latent_path = os.path.join(self.root_path, clip_name, 'latents_cogvideox', '1st_frame.pt')
        context_latents_path = [os.path.join(self.root_path, clip_name, 'latents_cogvideox', f'retrieval_{i+1}.pt') for i in range(3)]
        
        # masks
        anchor_mask_path = os.path.join(self.root_path, clip_name, 'masks', '1st_frame_mask.npz')
        context_masks_path = [os.path.join(self.root_path, clip_name, 'masks', f'retrieval_{i+1}_mask.npz') for i in range(3)]
        
        # plucker embedding
        # first_frame_camera_path = os.path.join(self.root_path.replace('Context_Anchor_funwarp', 'Wan_sample_720p_outputs'), clip_name, 'relative_cameras', f'1st_frame.npz')
        # context_camera_paths = [os.path.join(self.root_path.replace('Context_Anchor_funwarp', 'Wan_sample_720p_outputs'), clip_name, 'relative_cameras', f'retrieval_{i+1}.npz') for i in range(3)]
        first_frame_camera_path = os.path.join(self.root_path, clip_name, 'relative_cameras', f'1st_frame.npz')
        context_camera_paths = [os.path.join(self.root_path, clip_name, 'relative_cameras', f'retrieval_{i+1}.npz') for i in range(3)]
        
        video_caption_emb = torch.load(caption_emb_path, weights_only=True)
        
        source_latent = torch.load(source_latent_path, weights_only=True)[0].permute(1,0,2,3)
        anchor_latent_1st_frame = torch.load(anchor_latent_path, weights_only=True)[0].permute(1,0,2,3)
        anchor_latent_retrieved_frames = [torch.load(path, weights_only=True)[0].permute(1,0,2,3) for path in context_latents_path]
        anchor_latents = [anchor_latent_1st_frame] + anchor_latent_retrieved_frames

        video_reader = VideoReader(video_path)
        indices = [0]
        first_frame = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        first_frame = first_frame / 255.
        
        gt_and_first_frame_camera = torch.from_numpy(np.load(first_frame_camera_path)['pose'])
        context_cameras = [torch.from_numpy(np.load(path)['pose']) for path in context_camera_paths]
        
        video_length = len(video_reader)
        indices = np.linspace(0, video_length - 1, self.sample_n_frames, dtype=int)

        gt_and_first_frame_camera = gt_and_first_frame_camera[indices]
        context_cameras = [camera[indices] for camera in context_cameras]
        
        anchor_mask_1st_frame = np.load(anchor_mask_path)['mask'].astype(np.float16)
        anchor_masks_retrieved_frames = [np.load(path)['mask'].astype(np.float16) for path in context_masks_path]
        anchor_masks = [anchor_mask_1st_frame] + anchor_masks_retrieved_frames
        anchor_masks = torch.from_numpy(np.stack(anchor_masks))
        anchor_masks = anchor_masks[:,indices]
        
        return source_latent, anchor_latents, anchor_masks, first_frame, gt_and_first_frame_camera, context_cameras, video_caption_emb, clip_name
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        while True:
            try:
                source_latent, anchor_latents, anchor_masks, first_frame, gt_and_first_frame_camera, context_cameras, video_caption_emb, clip_name = self.get_batch(idx)
                break
            
            except Exception as e:
                idx = random.randint(0, self.length - 1)
        
        anchor_masks = self.pixel_transforms[0](anchor_masks)
        for transform in self.pixel_transforms:
            first_frame = transform(first_frame)
        
        data = {
            'source_latent': source_latent,
            'anchor_latents': anchor_latents,
            'anchor_masks': anchor_masks,
            'first_frame': first_frame,
            'gt_and_first_frame_camera': gt_and_first_frame_camera,
            'context_cameras': context_cameras,
            'caption_emb': video_caption_emb, 
            'clip_name': clip_name
        }
        return data