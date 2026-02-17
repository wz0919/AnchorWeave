import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def apply_tube_mask(
    controlnet_input_states: torch.Tensor,  # [B, # anchors, F, C, H, W]
    anchor_masks: torch.Tensor,  # [B, # anchors, F, 1, H_mask, W_mask]
    mask_ratio: float = 0.5,
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply tube mask: for each anchor, all frames use the same spatial mask.
    Mask is generated based on anchor_masks shape, then resized to controlnet_input_states shape.
    
    Args:
        controlnet_input_states: [B, # anchors, F, C, H, W] (e.g., 60x90 for latents)
        anchor_masks: [B, # anchors, F, 1, H_mask, W_mask] (e.g., 30x45)
        mask_ratio: Ratio of spatial area to mask (0.0 to 1.0)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        masked_controlnet_input_states: [B, # anchors, F, C, H, W]
        updated_anchor_masks: [B, # anchors, F, 1, H_mask, W_mask]
    """
    B, N, num_frames, C, H, W = controlnet_input_states.shape
    _, _, _, _, H_mask, W_mask = anchor_masks.shape
    device = controlnet_input_states.device
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Generate random spatial mask for each anchor (same across all frames)
    # mask: 0=mask, 1=not mask
    # Generate mask based on anchor_masks shape: [B, N, 1, 1, H_mask, W_mask]
    num_pixels = H_mask * W_mask
    num_mask_pixels = int(num_pixels * mask_ratio)
    
    tube_masks_small = torch.ones(B, N, 1, 1, H_mask, W_mask, device=device)  # Initialize to 1 (not mask)
    for b in range(B):
        for n in range(N):
            flat_mask = torch.ones(H_mask * W_mask, device=device)
            mask_indices = torch.randperm(H_mask * W_mask, device=device)[:num_mask_pixels]
            flat_mask[mask_indices] = 0.0  # Set selected pixels to 0 (mask)
            tube_masks_small[b, n, 0, 0] = flat_mask.reshape(H_mask, W_mask)
    
    # Expand mask to all frames: [B, N, num_frames, 1, H_mask, W_mask]
    tube_masks_small = tube_masks_small.expand(-1, -1, num_frames, -1, -1, -1)
    
    # Ensure first frame (frame 0) is never masked (I2V first frame condition)
    tube_masks_small[:, :, 0:1, :, :, :] = 1.0  # Set to 1 (not mask)
    
    # Resize mask to controlnet_input_states shape: [B, N, num_frames, 1, H, W]
    tube_masks = F.interpolate(
        tube_masks_small.reshape(B * N * num_frames, 1, H_mask, W_mask),
        size=(H, W),
        mode='nearest',
    ).reshape(B, N, num_frames, 1, H, W)
    
    # Apply mask to controlnet_input_states (0=mask, 1=not mask)
    masked_controlnet_input_states = controlnet_input_states * tube_masks
    
    # Update anchor_masks (combine with existing masks, use min to keep masked regions)
    updated_anchor_masks = anchor_masks * tube_masks_small
    
    return masked_controlnet_input_states, updated_anchor_masks


def apply_frame_mask(
    controlnet_input_states: torch.Tensor,  # [B, # anchors, F, C, H, W]
    anchor_masks: torch.Tensor,  # [B, # anchors, F, 1, H_mask, W_mask]
    mask_ratio: float = 0.3,
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply frame mask: for each anchor, randomly sample some frames and mask them entirely.
    Mask is generated based on anchor_masks shape, then resized to controlnet_input_states shape.
    
    Args:
        controlnet_input_states: [B, # anchors, F, C, H, W] (e.g., 60x90 for latents)
        anchor_masks: [B, # anchors, F, 1, H_mask, W_mask] (e.g., 30x45)
        mask_ratio: Ratio of frames to mask (0.0 to 1.0)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        masked_controlnet_input_states: [B, # anchors, F, C, H, W]
        updated_anchor_masks: [B, # anchors, F, 1, H_mask, W_mask]
    """
    B, N, num_frames, C, H, W = controlnet_input_states.shape
    _, _, _, _, H_mask, W_mask = anchor_masks.shape
    device = controlnet_input_states.device
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Generate frame mask for each anchor independently
    # mask: 0=mask, 1=not mask
    # Exclude first frame (frame 0) from masking (I2V first frame condition)
    num_mask_frames = max(1, int((num_frames - 1) * mask_ratio))  # Exclude first frame from count
    
    frame_masks = torch.ones(B, N, num_frames, 1, 1, 1, device=device)  # Initialize to 1 (not mask)
    for b in range(B):
        for n in range(N):
            available_frames = torch.arange(1, num_frames, device=device)
            mask_frame_indices = available_frames[torch.randperm(len(available_frames), device=device)[:num_mask_frames]]
            frame_masks[b, n, mask_frame_indices, 0, 0, 0] = 0.0  # Set to 0 (mask)
    
    # Expand mask to anchor_masks spatial dimensions: [B, N, num_frames, 1, H_mask, W_mask]
    frame_masks_small = frame_masks.expand(-1, -1, -1, -1, H_mask, W_mask)
    
    # Resize mask to controlnet_input_states shape: [B, N, num_frames, 1, H, W]
    frame_masks_large = F.interpolate(
        frame_masks_small.reshape(B * N * num_frames, 1, H_mask, W_mask),
        size=(H, W),
        mode='nearest',
    ).reshape(B, N, num_frames, 1, H, W)
    
    # Apply mask to controlnet_input_states (0=mask, 1=not mask)
    masked_controlnet_input_states = controlnet_input_states * frame_masks_large
    
    # Update anchor_masks (combine with existing masks, use min to keep masked regions)
    updated_anchor_masks = anchor_masks * frame_masks_small
    
    return masked_controlnet_input_states, updated_anchor_masks


def apply_anchor_sampling(
    context_cameras: torch.Tensor,  # [B, # anchors, F, 3, 4]
    controlnet_input_states: torch.Tensor,  # [B, # anchors, F, C, H, W]
    anchor_masks: torch.Tensor,  # [B, # anchors, F, 1, H, W]
    num_anchors: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a subset of anchors from the available anchors.
    
    Args:
        context_cameras: [B, # anchors, F, 3, 4]
        controlnet_input_states: [B, # anchors, F, C, H, W]
        anchor_masks: [B, # anchors, F, 1, H, W]
        num_anchors: Number of anchors to sample (None means sample randomly between 1 and original)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        sampled_context_cameras: [B, num_sampled_anchors, F, 3, 4]
        sampled_controlnet_input_states: [B, num_sampled_anchors, F, C, H, W]
        sampled_anchor_masks: [B, num_sampled_anchors, F, 1, H, W]
    """
    B, N, num_frames = context_cameras.shape[:3]
    _, _, _, C, H, W = controlnet_input_states.shape
    device = context_cameras.device
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Determine number of anchors to sample
    if num_anchors is None:
        num_anchors = torch.randint(1, N + 1, (1,), device=device).item()
    else:
        num_anchors = min(num_anchors, N)
    
    # Sample anchor indices (same for all batches)
    indices = torch.randperm(N, device=device)[:num_anchors]  # [num_anchors]
    
    # Sample from each tensor using direct indexing
    sampled_context_cameras = context_cameras[:, indices]  # [B, num_anchors, F, 3, 4]
    sampled_controlnet_input_states = controlnet_input_states[:, indices]  # [B, num_anchors, F, C, H, W]
    sampled_anchor_masks = anchor_masks[:, indices]  # [B, num_anchors, F, 1, H, W]
    
    return sampled_context_cameras, sampled_controlnet_input_states, sampled_anchor_masks


def apply_first_anchor_frame_mask(
    context_cameras: torch.Tensor,  # [B, # anchors, F, 3, 4]
    controlnet_input_states: torch.Tensor,  # [B, # anchors, F, C, H, W]
    anchor_masks: torch.Tensor,  # [B, # anchors, F, 1, H_mask, W_mask]
    min_mask_ratio: float = 0.25,
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Use only the first anchor and mask frames from a random ratio onwards.
    Returns only the first anchor's data.
    Note: context_cameras is not masked, only controlnet_input_states and anchor_masks are masked.
    Mask is generated based on anchor_masks shape, then resized to controlnet_input_states shape.
    
    Args:
        context_cameras: [B, # anchors, F, 3, 4]
        controlnet_input_states: [B, # anchors, F, C, H, W] (e.g., 60x90 for latents)
        anchor_masks: [B, # anchors, F, 1, H_mask, W_mask] (e.g., 30x45)
        min_mask_ratio: Minimum ratio to start masking from (default 0.25, meaning mask from 25% onwards)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        context_cameras: [B, 1, F, 3, 4] (first anchor only, unchanged)
        masked_controlnet_input_states: [B, 1, F, C, H, W] (first anchor only, masked)
        updated_anchor_masks: [B, 1, F, 1, H_mask, W_mask] (first anchor only, masked)
    """
    B, N, num_frames, C, H, W = controlnet_input_states.shape
    _, _, _, _, H_mask, W_mask = anchor_masks.shape
    device = controlnet_input_states.device
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Extract first anchor only
    first_anchor_states = controlnet_input_states[:, 0:1]  # [B, 1, num_frames, C, H, W]
    first_anchor_masks = anchor_masks[:, 0:1]  # [B, 1, num_frames, 1, H_mask, W_mask]
    first_anchor_cameras = context_cameras[:, 0:1]  # [B, 1, num_frames, 3, 4]
    
    mask_ratio = torch.rand(1, device=device).item() * (0.75 - min_mask_ratio) + min_mask_ratio
    mask_start_frame = int(num_frames * mask_ratio)
    # Ensure at least frame 1 is preserved (first frame condition)
    mask_start_frame = max(1, mask_start_frame)
    
    # Create frame mask based on anchor_masks shape: [B, 1, num_frames, 1, H_mask, W_mask]
    # mask: 0=mask, 1=not mask
    # Ensure first frame (frame 0) is never masked (I2V first frame condition)
    frame_masks_small = torch.ones(B, 1, num_frames, 1, H_mask, W_mask, device=device)  # Initialize to 1 (not mask)
    frame_masks_small[:, :, mask_start_frame:, :, :, :] = 0.0  # Set to 0 (mask)
    
    # Resize mask to controlnet_input_states shape: [B, 1, num_frames, 1, H, W]
    frame_masks_large = F.interpolate(
        frame_masks_small.reshape(B * 1 * num_frames, 1, H_mask, W_mask),
        size=(H, W),
        mode='nearest',
    ).reshape(B, 1, num_frames, 1, H, W)
    
    # Apply mask to controlnet_input_states (0=mask, 1=not mask)
    masked_controlnet_input_states = first_anchor_states * frame_masks_large
    
    # Update anchor_masks (combine with existing masks, use min to keep masked regions)
    updated_anchor_masks = first_anchor_masks * frame_masks_small
    
    # context_cameras remains unchanged (first anchor only)
    return first_anchor_cameras, masked_controlnet_input_states, updated_anchor_masks
