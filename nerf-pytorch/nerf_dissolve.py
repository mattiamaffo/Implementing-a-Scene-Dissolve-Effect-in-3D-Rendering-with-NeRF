import torch
import torch.nn.functional as F
import torch
import numpy as np
import scipy.interpolate
from perlin_noise import PerlinNoise

def interpolate_render_poses(render_poses, desired_frames, device):
    """
    Function to interpolate render poses to increase the number of frames.

    Args:
        render_poses (torch.Tensor): The original camera-to-world transformation matrices.
        desired_frames (int): The target number of frames after interpolation.
        device (torch.device): The device where the tensor should be moved (e.g., "cuda" or "cpu").

    Returns:
        torch.Tensor: The interpolated render poses.
    """

    render_poses_cpu = render_poses.cpu().numpy()

    # Generate frame indices for interpolation
    frame_indices = np.linspace(0, len(render_poses) - 1, num=desired_frames)

    # Interpolate camera poses using scipy
    interp_func = scipy.interpolate.interp1d(
        np.arange(len(render_poses)), render_poses_cpu, axis=0, kind='linear'
    )
    
    interpolated_poses = torch.tensor(interp_func(frame_indices), dtype=torch.float32, device=device)

    return interpolated_poses

def radial_dissolve(raw, pts, frame_idx, total_frames, scene_center=(0, 0, 0)):
    """
    Radial dissolve effect (from center outward).
    
    - Starts with full visibility at the center.
    - Progressively fades out towards the edges.
    - Only radial distance determines transparency.

    Args:
        raw: torch.Tensor, raw output from NeRF. Shape: [N_rays, N_samples, 4]
        pts: torch.Tensor, 3D points sampled along the rays. Shape: [N_rays, N_samples, 3]
        frame_idx: int, current frame index.
        total_frames: int, total number of frames.
        scene_center: tuple, (x, y, z) coordinates of the scene center.

    Returns:
        sigma: torch.Tensor, modified density with radial dissolve.
        dissolve_mask: torch.Tensor, dissolve effect visualization.
    """
    # distances from the scene center
    scene_center = torch.tensor(scene_center, dtype=torch.float32, device=pts.device)
    distances = torch.norm(pts - scene_center, dim=-1)  # Shape: [N_rays, N_samples]

    # Normalize distances to range [0,1]
    max_dist = distances.max().clamp(min=1.0)
    normalized_distances = distances / max_dist

    dissolve_threshold = (frame_idx / total_frames) ** 1.5  # Controls dissolve expansion -> velocity

    # Sigmoid with inverted threshold for radial dissolve
    dissolve_mask = torch.sigmoid(20 * (normalized_distances - dissolve_threshold))

    sigma = raw[..., 3] * dissolve_mask  # Shape: [N_rays, N_samples]

    sigma[sigma < 0.02] = 0  # Reduce noise

    return sigma, dissolve_mask

def galaxy_swirl_dissolve(raw, pts, frame_idx, total_frames, scale=5, noise_intensity=0.7):
    """
    Galaxy Swirl Dissolve:
    - Combines a "galactic swirl" effect with Perlin noise.
    - The object dissolves in a spiral-like manner, as if being sucked into a galactic black hole.
    - The effect starts from the "galactic center" and spirals outwards with colorful swirls.

    Args:
        raw: torch.Tensor, raw output from NeRF, shape [N_rays, N_samples, 4]
        pts: torch.Tensor, 3D points sampled along the rays, shape [N_rays, N_samples, 3]
        frame_idx: int, current frame index
        total_frames: int, total number of frames
        scale: int, scale factor for Perlin noise
        noise_intensity: float, controls the intensity of the Perlin noise

    Returns:
        sigma: torch.Tensor, modified density with swirl effect.
        dissolve_mask: torch.Tensor, dissolve mask for debugging.
    """

    # Normalize 3D positions to range [0,1]
    min_vals, max_vals = pts.amin(dim=0), pts.amax(dim=0)
    random_offset = torch.rand(3, device=pts.device) * 0.3  # offset nel range [0,0.3]
    norm_pts = (pts - min_vals) / (max_vals - min_vals + 1e-8)
    norm_pts = norm_pts + random_offset
    norm_pts = norm_pts % 1.0  # rimappiamo in [0,1]


    # Perlin noise generation on a grid (without the method for optimization)
    grid_size = 512
    noise_grid = torch.rand((1, 1, grid_size, grid_size, grid_size), device=pts.device)

    # Interpolate the values using grid_sample
    sample_coords = norm_pts.view(1, 1, *pts.shape[:2], 3) * 2 - 1
    perlin_mask = F.grid_sample(
    noise_grid, 
    sample_coords,
    mode="bilinear",
    align_corners=False,
    padding_mode="reflection"  # O "border"
        ).squeeze()

    # Define the radial distance from the center
    # We consider the center to be at (0.5, 0.5, 0.5) in normalized coordinates
    x = norm_pts[..., 0] - 0.5
    z = norm_pts[..., 2] - 0.5

    # Compute the radial distance and angle in polar coordinates
    r = torch.sqrt(x**2 + z**2)
    angle = torch.atan2(z, x)

    # Create a swirling effect based on the angle
    # More intense swirls -> higher swirl_scale
    swirl_scale = 4.0
    swirl_offset = torch.sin(angle * swirl_scale + frame_idx * 0.1) * 0.1
    radial_offset = r * 0.3

    # Define the dissolve threshold based on the frame index
    # Start slowly and increase the dissolve speed over time, adding noise and swirls
    dissolve_threshold = (frame_idx / total_frames) ** 1.5
    dissolve_threshold += (perlin_mask * noise_intensity)
    dissolve_threshold += swirl_offset + radial_offset

    dissolve_mask = torch.sigmoid(8 * (perlin_mask - dissolve_threshold))

    # Apply the dissolve effect to the density
    sigma = raw[..., 3] * dissolve_mask
    sigma = torch.where(sigma < 0.02, torch.tensor(0.0, device=sigma.device), sigma)

    return sigma, dissolve_mask

def window_dissolve(raw, pts, frame_idx, total_frames, spread=10, offset=0.05):
    """
    Applies a top-to-bottom dissolve effect based on the Y-coordinate.

    - Starts fully visible at the top.
    - Gradually dissolves downwards.
    - The process is smooth and continuous without inversions.

    Args:
        raw (torch.Tensor): Raw output from NeRF. Shape: [N_rays, N_samples, 4]
        pts (torch.Tensor): 3D points sampled along the rays. Shape: [N_rays, N_samples, 3]
        frame_idx (int): Current frame index.
        total_frames (int): Total number of frames.
        spread (float): Controls the speed of the dissolve.
        offset (float): Prevents too sharp a dissolve.

    Returns:
        sigma (torch.Tensor): Modified density with top-bottom dissolve.
        dissolve_mask (torch.Tensor): Dissolve mask for debugging.
    """

    # Normalize Y-coordinates to [0,1]
    min_y, max_y = pts[..., 1].min(), pts[..., 1].max()
    normalized_y = (pts[..., 1] - min_y) / (max_y - min_y)

    # Define progressive dissolve threshold
    dissolve_threshold = (frame_idx / total_frames) ** 1.3 
    dissolve_threshold = max(0, min(1, dissolve_threshold))

    # Compute dissolve mask based on Y-coordinates with sigmoid
    dissolve_mask = torch.sigmoid(spread * (normalized_y - dissolve_threshold + offset))

    sigma = raw[..., 3] * dissolve_mask

    sigma[sigma < 0.02] = 0  

    return sigma, dissolve_mask

def perlin_dissolve(raw, pts, frame_idx, total_frames, scale=5, noise_intensity=0.7):
    """
    Applies an optimized Perlin noise-based dissolve effect using GPU acceleration.
    
    Args:
        raw: torch.Tensor, raw output from NeRF. Shape: [N_rays, N_samples, 4]
        pts: torch.Tensor, 3D points sampled along the rays. Shape: [N_rays, N_samples, 3]
        frame_idx: int, current frame index.
        total_frames: int, total number of frames.
        scale: int, scale factor for Perlin noise (higher = larger smooth areas, lower = more detail).
        noise_intensity: float, controls the strength of the Perlin noise effect.

    Returns:
        sigma: torch.Tensor, modified density with Perlin dissolve.
        dissolve_mask: torch.Tensor, dissolve effect visualization.
    """

    # Normalize 3D positions to range [0,1]
    min_vals, max_vals = pts.amin(dim=0), pts.amax(dim=0)
    norm_pts = (pts - min_vals) / (max_vals - min_vals) 

    # Generate Perlin noise on a grid
    grid_size = 128 
    noise_grid = torch.rand((1, 1, grid_size, grid_size, grid_size), device=pts.device)

    # Interpolate the values using grid_sample
    sample_coords = norm_pts.view(1, 1, *pts.shape[:2], 3) * 2 - 1 
    perlin_mask = F.grid_sample(noise_grid, sample_coords, mode="bilinear", align_corners=True).squeeze()
    perlin_mask = perlin_mask.view(raw[..., 3].shape)

    # Define progressive dissolve threshold
    dissolve_threshold = (frame_idx / total_frames) ** 1.5

    # Compute dissolve mask using Perlin noise pattern
    dissolve_mask = torch.sigmoid(10 * (perlin_mask - dissolve_threshold))

    sigma = raw[..., 3] * dissolve_mask

    sigma = torch.where(sigma < 0.05, torch.tensor(0.0, device=sigma.device), sigma)

    return sigma, dissolve_mask

def thanos_dissolve(raw, pts, frame_idx, total_frames, scale=5, noise_intensity=0.7):
    """
    Thanos Snap Dissolve Effect:
    - Gradual disintegration and dispersion effect similar to Thanos' snap.
    - Removed horizontal stripes by increasing the noise resolution.
    - Added vertical variation to avoid uniform bands.
    - Smoothed the dissolve effect with a more fluid distribution.

    Args:
        raw: torch.Tensor, raw output from NeRF. Shape: [N_rays, N_samples, 4]
        pts: torch.Tensor, 3D points sampled along the rays. Shape: [N_rays, N_samples, 3]
        frame_idx: int, current frame index.
        total_frames: int, total number of frames.
        scale: int, scale factor for Perlin noise (higher = larger smooth areas, lower = more detail).
        noise_intensity: float, controls the strength of the Perlin noise effect.

    Returns:
        sigma: torch.Tensor, modified density with Thanos dissolve effect.
        dissolve_mask: torch.Tensor, dissolve effect visualization. 
    """

    min_vals, max_vals = pts.amin(dim=0), pts.amax(dim=0)
    norm_pts = (pts - min_vals) / (max_vals - min_vals + 1e-8)

    # Generate Perlin noise on a grid
    grid_size = 256  # Increments the resolution to avoid horizontal stripes
    noise_grid = torch.rand((1, 1, grid_size, grid_size, grid_size), device=pts.device)

    sample_coords = norm_pts.view(1, 1, *pts.shape[:2], 3) * 2 - 1
    perlin_mask = F.grid_sample(noise_grid, sample_coords, mode="bilinear", align_corners=True).squeeze()

    # Add vertical variation to avoid uniform bands
    vertical_variation = torch.sin(norm_pts[..., 1] * 10 + frame_idx * 0.1) * 0.05
    dissolve_threshold = (frame_idx / total_frames) ** 1.5 + perlin_mask * noise_intensity + vertical_variation

    dissolve_mask = torch.sigmoid(10 * (perlin_mask - dissolve_threshold))

    sigma = raw[..., 3] * dissolve_mask
    sigma = torch.where(sigma < 0.02, torch.tensor(0.0, device=sigma.device), sigma)

    return sigma, dissolve_mask

def simple_dissolve(raw, rays_d, frame_idx, total_frames):
    """
    Function to apply a dissolve effect to the density (sigma).

    Args:
        raw (torch.Tensor): The raw output from the NeRF model.
        rays_d (torch.Tensor): Ray directions.
        frame_idx (int): Current frame index.
        total_frames (int): Total number of frames.

    Returns:
        torch.Tensor: Modified sigma with the dissolve effect applied.
        torch.Tensor: Fade factor for debugging purposes.
    """

    # Define exponential decay for the dissolve effect
    decay_rate = 4  # Higher values result in a faster dissolve
    fade_factor = torch.exp(torch.tensor(-decay_rate * (frame_idx / total_frames), dtype=torch.float32, device=rays_d.device))

    # Reduce density (sigma) to enforce transparency
    sigma = raw[...,3] * fade_factor
    sigma[sigma < 0.05] = 0

    return sigma, fade_factor
