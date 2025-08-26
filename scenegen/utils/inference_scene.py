from typing import *
import os
import time
import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image

from ..pipelines import SceneGenImageToScenePipeline

def split_rgb_mask(rgb_image, seg_image, order: str) -> List[Image.Image]:
    if isinstance(rgb_image, str):
        rgb_image = Image.open(rgb_image)
    if isinstance(seg_image, str):
        seg_image = Image.open(seg_image)
    rgb_image = rgb_image.convert("RGB")
    seg_image = seg_image.convert("L")

    rgb_array = np.array(rgb_image)
    seg_array = np.array(seg_image)

    label_ids = np.unique(seg_array)
    label_ids = label_ids[label_ids > 0]

    if order in ["smallest", "largest"]:
        # Sort label_ids by mask area (small to large)
        areas = []
        for segment_id in label_ids:
            area = np.sum(seg_array == segment_id)  # Count pixels in this segment
            areas.append((segment_id, area))
        
        areas.sort(key=lambda x: x[1])  # Sort by area

        if order == "largest":
            areas = areas[::-1]

        label_ids = [id for id, _ in areas]
    if order == "order":
        label_ids = sorted(label_ids)
    
    instance_rgbs, instance_masks, scene_rgbs = [], [], []

    for segment_id in label_ids:
        # Create an RGBA image with transparent background
        segment_rgba = np.zeros((rgb_array.shape[0], rgb_array.shape[1], 4), dtype=np.uint8)
        
        # Create the mask for this segment
        mask = np.zeros_like(seg_array, dtype=np.uint8)
        mask[seg_array == segment_id] = 255
        
        # Set RGB values where the mask is active
        segment_rgba[mask == 255, :3] = rgb_array[mask == 255]
        # Set alpha channel - transparent background (0), opaque foreground (255)
        segment_rgba[mask == 255, 3] = 255
        
        # Convert to PIL Image
        segment_rgba_image = Image.fromarray(segment_rgba, mode="RGBA")
        # Create a 3-channel mask
        mask_rgb = np.stack([mask, mask, mask], axis=2)
        segment_mask_image = Image.fromarray(mask_rgb, mode="RGB")
        
        instance_rgbs.append(segment_rgba_image)
        instance_masks.append(segment_mask_image)
        scene_rgbs.append(rgb_image)
    
    dir_name = f"tmp/{time.time()}"
    os.makedirs(dir_name, exist_ok=True)
    for i, (ir, im) in enumerate(zip(instance_rgbs, instance_masks)):
        ir.save(os.path.join(dir_name, f"{i}.png"))
        im.save(os.path.join(dir_name, f"{i}_mask.png"))
    
    scene_rgbs[0].save(os.path.join(dir_name, "scene.jpg"))
    combined_mask = (seg_array > 0).astype(np.uint8) * 255
    masked_rgba = np.zeros((rgb_array.shape[0], rgb_array.shape[1], 4), dtype=np.uint8)
    masked_rgba[combined_mask == 255, :3] = rgb_array[combined_mask == 255]
    masked_rgba[..., 3] = combined_mask
    Image.fromarray(masked_rgba, mode="RGBA").save(os.path.join(dir_name, "masked_scene.png"))

    return instance_rgbs, instance_masks, scene_rgbs


def run_scene(
    pipe: SceneGenImageToScenePipeline,
    rgb_image: Union[str, Image.Image],
    seg_image: Union[str, Image.Image],
    seed: int = 42,
    ss_num_inference_steps: int = 50,
    ss_cfg_strength: float = 3.0,
    ss_cfg_interval: List[float] = [0, 1.0],
    ss_rescale_t: float = 1.0,
    slat_num_inference_steps: int = 25,
    slat_cfg_strength: float = 5.0,
    slat_cfg_interval: List[float] = [0.5, 1.0],
    slat_rescale_t: float = 3.0,
    order: str = "order",
    positions_type: Literal['last', 'avg'] = 'last',
    simplify: float = 0.95,
    texture_size: int = 1024,
) -> trimesh.Scene:
    
    torch.manual_seed(seed)

    sparse_structure_sampler_params={
            "steps": ss_num_inference_steps,
            "cfg_strength": ss_cfg_strength,
            "cfg_interval": ss_cfg_interval,
            "rescale_t": ss_rescale_t
        }
    
    slat_sampler_params={
            "steps": slat_num_inference_steps,
            "cfg_strength": slat_cfg_strength,
            "cfg_interval": slat_cfg_interval,
            "rescale_t": slat_rescale_t
        }
    
    if len(rgb_image) == 1 and len(seg_image) == 1:
        rgb_image = rgb_image[0]
        seg_image = seg_image[0]

        instance_rgbs, instance_masks, scene_rgbs = split_rgb_mask(rgb_image, seg_image, order=order)
        print(f"Number of instances: {len(instance_rgbs) and len(instance_masks)}")
        
        results = pipe.run_scene(
            image=instance_rgbs,
            mask_image=instance_masks,
            scene_image=scene_rgbs[0],
            preprocess_image=True,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            slat_sampler_params=slat_sampler_params,
            positions_type=positions_type,
            simplify=simplify,
            texture_size=texture_size,
        )
        scene = results["scene"]
    elif len(rgb_image) > 1 and len(seg_image) > 1:
        instance_rgbs, instance_masks, scene_rgbs = [], [], []
        for rgb, seg in zip(rgb_image, seg_image):
            ir, im, sr = split_rgb_mask(rgb, seg, order="order")
            instance_rgbs.append(ir)
            instance_masks.append(im)
            scene_rgbs.append(sr[0])

        results = pipe.run_scene(
            image=instance_rgbs,
            mask_image=instance_masks,
            scene_image=scene_rgbs,
            preprocess_image=True,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            slat_sampler_params=slat_sampler_params,
            positions_type=positions_type,
            simplify=simplify,
            texture_size=texture_size,
        )
        scene = results["scene"]

    return scene