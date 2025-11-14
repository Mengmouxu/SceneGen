import os
import torch
import json
import math
import logging
import trimesh
from PIL import Image, ImageChops
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from torchvision.io import read_image
import torchvision

from .base_system import BaseSystem
import time
from tqdm import tqdm
from metrics import (
    compute_chamfer_distance,
    compute_fscore,
    compute_volume_iou,
    normalize_points,
    sample_points_from_meshes,
    point_alignment,
    compute_fid,
    load_clip_model,
    load_dinov2_model,
    load_lpips_model,
    compute_render_metrics_,
)

logger = logging.getLogger(__name__)


class SceneSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):

        input_format: str = "glb"
        gt_format: str = "glb"

        gpu_num: int = 1
        gpu_id: int = 0
        
        num_points: int = 20480
        chunk_size: int = 10240
        
        eval_scene_level: bool = True
        eval_object_level: bool = True
        alignment_mode: str = 'filterreg'
        alignment_max_iterations: int = 50
        alignment_tolerance: float = 1e-5
        fscore_threshold: float = 0.1
        scene_alignment: bool = True
        scene_visualize: bool = False
        asset_alignment: bool = True
        asset_visualize: bool = False

        render_eval: bool = True
        dataset_dir: str = "3D-FUTURE"
        set: str = "test"
        blender_path: str = "blender-3.0.1-linux-x64"
        clip_model_name: str = "ViT-L/14"
        dinov2_model_name: str = "dinov2_vitl14_reg"
        lpips_model_name: str = "alex"
        masked_scene_images_eval: bool = True
        render_gt_images_eval: bool = True
        
        metrics: List[str] = field(default_factory=lambda: [
            "scene_cd", "scene_fscore", "object_cd", "object_fscore", "iou_bbox",
            "psnr", "ssim", "lpips", "fid", "clip_similarity", "dinov2_similarity",
        ])
    
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.metrics_values = {}
        self.cfg.render_eval = self.cfg.render_eval and self.cfg.eval_scene_level
        if self.cfg.render_eval:
            self.scene_json_path = os.path.join(self.cfg.dataset_dir, 'raw', '3D-FUTURE-scene', 'GT', f'{self.cfg.set}_set.json')
            self.scene_image_path = os.path.join(self.cfg.dataset_dir, f'masked_images_{self.cfg.set}')
            self.scene_render_gt_path = self.cfg.output_dir + "_render"
            self.scene_render_pred_path = self.cfg.input_dir + "_render"
            self.scene_render_config_path =  os.path.join(self.scene_render_gt_path, 'render_cfg.json')
            assert os.path.exists(self.scene_json_path), f"Scene JSON file {self.scene_json_path} does not exist."
            assert os.path.exists(self.scene_image_path), f"Scene image directory {self.scene_image_path} does not exist."
            assert os.path.exists(self.scene_render_gt_path), f"Scene render ground truth directory {self.scene_render_gt_path} does not exist."
            assert os.path.exists(self.scene_render_config_path), f"Scene render config file {self.scene_render_config_path} does not exist."
            os.makedirs(self.scene_render_pred_path, exist_ok=True)

            self.scene_render_config = json.load(open(self.scene_render_config_path, 'r'))

            logger.info(f"Loading CLIP model {self.cfg.clip_model_name} and DINOv2 model {self.cfg.dinov2_model_name}")
            self.clip_model, self.clip_preprocess = load_clip_model(self.cfg.clip_model_name)
            self.dinov2_model, self.dinov2_preprocess = load_dinov2_model(self.cfg.dinov2_model_name)
            self.lpips_model = load_lpips_model(self.cfg.lpips_model_name)

            self.clip_model = self.clip_model.to(self.device).eval()
            self.dinov2_model = self.dinov2_model.to(self.device).eval()
            self.lpips_model = self.lpips_model.to(self.device).eval()

            self.fid_pred_images = []
            self.fid_gt_render_images = []
            self.fid_gt_scene_images = []

        assert cfg.alignment_mode in ['filterreg', 'cpd']
    
    def setup(self):
        super().setup()
        logger.info(f"Number of points sampled from each asset mesh: {self.cfg.num_points}")

    @torch.no_grad()
    def compute_metrics(
        self,
        pred_mesh_list: List[trimesh.Trimesh],
        gt_mesh_list: List[trimesh.Trimesh],
        basename: str = None,
        pred_path: Optional[str] = None,
        render_cfg: Optional[Dict] = None,
    ):
        
        num_instances = len(pred_mesh_list)
        scene_vertices_pred = []
        scene_vertices_gt = []
        object_vertices_pred = []
        object_vertices_gt = []

        for mesh in pred_mesh_list:
            points = sample_points_from_meshes(mesh, self.cfg.num_points)
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points)
            scene_vertices_pred.append(points.to(self.device))
            object_vertices_pred.append(points.to(self.device))
        scene_vertices_pred = torch.cat(scene_vertices_pred) if len(scene_vertices_pred) > 1 else scene_vertices_pred[0]
        for mesh in gt_mesh_list:
            points = sample_points_from_meshes(mesh, self.cfg.num_points)
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points)
            scene_vertices_gt.append(points.to(self.device))
            object_vertices_gt.append(points.to(self.device))
        scene_vertices_gt = torch.cat(scene_vertices_gt) if len(scene_vertices_gt) > 1 else scene_vertices_gt[0]

        assert scene_vertices_pred.dim() == 3 and scene_vertices_gt.dim() == 3, f"The dimension of vertices should be 3, but got {scene_vertices_pred.dim()} and {scene_vertices_gt.dim()}"
        metrics = {}

        if self.cfg.eval_scene_level:
            scene_metrics, R, T = self._compute_scene_metrics(
                scene_vertices_pred,
                scene_vertices_gt,
                num_instances,
                use_alignment=self.cfg.scene_alignment,
                basename=basename,
                visualize=self.cfg.scene_visualize,
            )
            metrics.update(scene_metrics)
            if self.cfg.render_eval and render_cfg is not None:
                render_metrics = self.compute_render_metrics(
                    pred_path,
                    render_cfg,
                    R=R,
                    T=T,
                )
                metrics.update(render_metrics)
        
        if self.cfg.eval_object_level:
            object_metrics = self._compute_object_metrics(
                scene_vertices_pred,
                scene_vertices_gt,
                num_instances,
                use_alignment=self.cfg.asset_alignment,
                basename=basename,
                visualize=self.cfg.asset_visualize,
            )
            metrics.update(object_metrics)
        
        return metrics

    def _compute_scene_metrics(
            self,
            vertices_pred,
            vertices_gt,
            num_instances,
            use_alignment: bool = True,
            basename: Optional[str] = None,
            visualize: bool = False,
        ):
        vertices_scene_pred = vertices_pred.reshape(-1, 3).unsqueeze(0)
        vertices_scene_gt = vertices_gt.reshape(-1, 3).unsqueeze(0)
        
        if use_alignment:
            vertices_scene_pred, R, T = point_alignment(
                vertices_scene_pred, 
                vertices_scene_gt,
                max_iterations=self.cfg.alignment_max_iterations,
                mode=self.cfg.alignment_mode,
            )

        if visualize:
            if basename is not None:
                basename = basename + "_scene_" + self.cfg.filename
            self.visualize_point_cloud(vertices_scene_pred, vertices_scene_gt, basename)
        
        
        cds = compute_chamfer_distance(
            vertices_scene_pred, vertices_scene_gt, chunk_size=self.cfg.chunk_size
        )
        fscore_scene = compute_fscore(
            vertices_scene_pred, vertices_scene_gt, tau=self.cfg.fscore_threshold
        )
        
        # Use torch.split to efficiently partition the scene vertices into object-level vertices.
        object_vertices_pred = vertices_scene_pred[0].view(num_instances, self.cfg.num_points, 3)
        object_vertices_gt = vertices_scene_gt[0].view(num_instances, self.cfg.num_points, 3)

        iou_bbox = compute_volume_iou(
            object_vertices_pred, object_vertices_gt, mode="bbox"
        )
        
        return {
            "scene_cd": cds[0],
            "scene_cd_1": cds[1],
            "scene_cd_2": cds[2],
            "scene_fscore": fscore_scene,
            "iou_bbox": iou_bbox,
        }, R, T


    def compute_render_metrics(
            self,
            pred_path: str,
            render_cfg: Dict,
            R: Optional[torch.Tensor] = None,
            T: Optional[torch.Tensor] = None,
    ):
        metrics = {}
        render_path = os.path.join(self.scene_render_pred_path, os.path.basename(pred_path).replace('.glb', '.png'))

        if os.path.exists(render_path):
            try:
                pred_render_image = read_image(render_path).float().to(self.device) / 255.0
            except Exception as e:
                pred_render_image = None
        else:
            pred_render_image = None

        if pred_render_image is None:
            R = R[0].cpu().numpy()
            T = T[0].cpu().numpy()

            pred_scene = trimesh.load(pred_path)

            if R is not None and T is not None:
                # Apply rotation and translation to the mesh
                transform = np.eye(4, dtype=np.float32)
                transform[:3, :3] = R
                transform[:3, 3] = T.flatten()
                pred_scene.apply_transform(transform)

            pred_tmp_path = os.path.join(".cache", os.path.basename(pred_path).split('.')[0] + "_tmp."+ self.cfg.input_format)
            os.makedirs(os.path.dirname(pred_tmp_path), exist_ok=True)
            pred_scene.export(pred_tmp_path)

            width = render_cfg.get('width', 1200)
            height = render_cfg.get('height', 1200)
            fov = render_cfg.get('fov', math.radians(60))
            camera_pos = render_cfg.get('camera_pos', (0, 0, 0))
            camera_rotation = render_cfg.get('camera_rotation', (0, 0, 0))
            blender_script = os.path.join(".cache", f"temp_render_script_{self.cfg.gpu_id}.py")
            with open(blender_script, 'w') as f:
                f.write(f"""
import bpy
import math
import sys

# Clear existing objects
bpy.ops.wm.read_factory_settings(use_empty=True)
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

bpy.ops.import_scene.gltf(filepath="{pred_tmp_path}")

# Set up rendering
scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# --- GPU Acceleration Setup ---
scene.cycles.device = 'GPU'
prefs = bpy.context.preferences
cycles_prefs = prefs.addons['cycles'].preferences

# Attempt to set compute device to OPTIX, fall back to CUDA, then HIP, then CPU
if hasattr(cycles_prefs, 'compute_device_type'):
    try:
        cycles_prefs.compute_device_type = 'OPTIX'
    except TypeError:
        try:
            cycles_prefs.compute_device_type = 'CUDA'
        except TypeError:
            try:
                cycles_prefs.compute_device_type = 'HIP'
            except TypeError:
                scene.cycles.device = 'CPU'

# Enable all available GPUs of the selected type
cycles_prefs.get_devices()
for device in cycles_prefs.devices:
    if device.type == cycles_prefs.compute_device_type:
        device.use = True

# Optimize render settings for speed
scene.cycles.samples = 128
scene.cycles.use_denoising = True
if hasattr(scene.cycles, 'denoiser'):
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 2
scene.cycles.transparent_max_bounces = 2
scene.cycles.transmission_bounces = 2
scene.cycles.volume_bounces = 0

scene.render.resolution_x = {width}
scene.render.resolution_y = {height}
scene.render.resolution_percentage = 100
scene.render.film_transparent = True # Set to True to render with a transparent background

# Set render output format to PNG with RGBA
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# Set up the camera
cam_data = bpy.data.cameras.new('Camera')
cam_obj = bpy.data.objects.new('Camera', cam_data)
bpy.context.scene.collection.objects.link(cam_obj)

# Set camera FOV
cam_data.angle = {fov}

# Position camera
cam_obj.location = {camera_pos}
cam_obj.rotation_mode = 'XYZ'
cam_obj.rotation_euler = [math.radians(d) for d in {camera_rotation}]

# Set the camera as the active camera
bpy.context.scene.camera = cam_obj

# Set up lighting from multiple directions for full illumination
def add_light(name, location, energy, light_type='POINT'):
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_data.energy = energy
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location

# Add multiple point lights from different positions to ensure even lighting
add_light("Point_Front_Top_Right", (4, -4, 4), 2000)
add_light("Point_Front_Top_Left", (-4, -4, 4), 2000)
add_light("Point_Back_Top_Right", (4, 4, 4), 1500)
add_light("Point_Back_Top_Left", (-4, 4, 4), 1500)
add_light("Point_Bottom", (0, 0, -5), 1000)

# Render
scene.render.filepath = "{render_path}"
bpy.ops.render.render(write_still=True)
        """)
            
            # Call Blender to render the scene
            blender_executable = os.path.join(self.cfg.blender_path, "blender") if not self.cfg.blender_path.endswith("blender") else self.cfg.blender_path
            blender_cmd = f"{blender_executable} --background --python {blender_script} > /dev/null 2>&1"
            os.system(blender_cmd)
            
            # Clean up the temporary script
            os.remove(blender_script)
            os.remove(pred_tmp_path)

            pred_render_image = read_image(render_path).to(self.device).float() / 255.0

        if self.cfg.render_gt_images_eval:
            gt_render_image_path = os.path.join(self.scene_render_gt_path, os.path.basename(pred_path).replace('.glb', '.png'))
            gt_render_image = read_image(gt_render_image_path).float().to(self.device) / 255.0

            # Convert RGBA to RGB by compositing with a white background
            if pred_render_image.shape[0] == 4:
                alpha_channel = pred_render_image[3:4, :, :]
                white_bg_pred = torch.ones_like(pred_render_image[:3, :, :])
                pred_render_image = alpha_channel * pred_render_image[:3, :, :] + (1 - alpha_channel) * white_bg_pred
            
            if gt_render_image.shape[0] == 4:
                alpha_channel = gt_render_image[3:4, :, :]
                white_bg_gt = torch.ones_like(gt_render_image[:3, :, :])
                gt_render_image = alpha_channel * gt_render_image[:3, :, :] + (1 - alpha_channel) * white_bg_gt

            self.fid_pred_images.append(render_path)
            self.fid_gt_render_images.append(gt_render_image_path)
            
            metrics.update(
                compute_render_metrics_(
                    pred_render_image,
                    gt_render_image,
                    self.lpips_model,
                    self.clip_model,
                    self.clip_preprocess,
                    self.dinov2_model,
                    self.dinov2_preprocess,
                    name="render"
                )
            )
        
        if self.cfg.masked_scene_images_eval:
            masked_scene_image_path = os.path.join(self.scene_image_path, os.path.basename(pred_path).split('.')[0], 'scene.jpg')    
            masked_scene_image_dir = os.path.dirname(masked_scene_image_path)
            save_path = os.path.join(masked_scene_image_dir, 'masked_scene.png')
            segmented_image = None
            if os.path.exists(save_path):
                try:
                    segmented_image = read_image(save_path).float().to(self.device) / 255.0
                except Exception as e:
                    segmented_image = None
            if segmented_image is None:
                masked_scene_image = read_image(masked_scene_image_path).float().to(self.device) / 255.0

                # Create a composite mask from all mask files in the directory
                composite_mask = torch.zeros((masked_scene_image.shape[1], masked_scene_image.shape[2]), device=self.device)
                for f in os.listdir(masked_scene_image_dir):
                    if '_mask' in f and f.endswith(('.png', '.jpg', '.jpeg')):
                        mask_path = os.path.join(masked_scene_image_dir, f)
                        mask_img = read_image(mask_path).float().to(self.device) / 255.0
                        if mask_img.shape[1:] == composite_mask.shape:
                            composite_mask = torch.maximum(composite_mask, mask_img[0])

                # Apply the composite mask to the alpha channel of the scene image
                segmented_image = masked_scene_image.clone()
                alpha_channel = composite_mask.unsqueeze(0)
                segmented_image = torch.cat((segmented_image[:3], alpha_channel), dim=0)

                # Optional: save the generated masked image for debugging
                torchvision.utils.save_image(segmented_image.cpu(), save_path)

            # Convert RGBA to RGB by compositing with a white background
            if segmented_image.shape[0] == 4:
                alpha_channel = segmented_image[3:4]
                white_bg = torch.ones_like(segmented_image[:3])
                segmented_image = alpha_channel * segmented_image[:3] + (1 - alpha_channel) * white_bg

            self.fid_gt_scene_images.append(save_path)
            
            metrics.update(
                compute_render_metrics_(
                    pred_render_image,
                    segmented_image,
                    self.lpips_model,
                    self.clip_model,
                    self.clip_preprocess,
                    self.dinov2_model,
                    self.dinov2_preprocess,
                    name="scene"
                )
            )

        return metrics

    def _compute_object_metrics(
            self,
            vertices_pred,
            vertices_gt,
            num_instances,
            use_alignment: bool = True,
            basename: Optional[str] = None,
            visualize: bool = False,
        ):
        assert vertices_pred.dim() == 3 and vertices_gt.dim() == 3, f"The dimension of vertices should be 3, but got {vertices_pred.dim()} and {vertices_gt.dim()}"
        assert vertices_pred.shape[0] == num_instances and vertices_gt.shape[0] == num_instances, f"The number of instances should be {num_instances}, but got {vertices_pred.shape[0]} and {vertices_gt.shape[0]}"

        object_vertices_pred = normalize_points(vertices_pred)
        object_vertices_gt = normalize_points(vertices_gt)

        if use_alignment:
            object_vertices_pred, _, _ = point_alignment(
                object_vertices_pred, 
                object_vertices_gt,
                max_iterations=self.cfg.alignment_max_iterations,
                mode=self.cfg.alignment_mode,
            )

        if visualize:
            if basename is not None:
                basename = basename + "_object_" + self.cfg.filename
            self.visualize_point_cloud(object_vertices_pred, object_vertices_gt, basename)
        
        # Compute metrics and immediately detach to free computation graph memory.
        cd_object = compute_chamfer_distance(
            object_vertices_pred, object_vertices_gt, chunk_size=self.cfg.chunk_size
        )[0].detach()

        fscore_object = compute_fscore(
            object_vertices_pred, object_vertices_gt, tau=self.cfg.fscore_threshold
        ).detach()
        
        return {
            "object_cd": cd_object,
            "object_fscore": fscore_object,
        }
    
    def load_mesh(self, file_path):
        try:
            mesh = trimesh.load(file_path)
            # Check if the loaded object is a scene (contains multiple meshes)
            if isinstance(mesh, trimesh.Scene):
                mesh_list = []
                # Get geometries with their transforms
                for node_name in mesh.graph.nodes_geometry:
                    transform = mesh.graph.get(node_name)[0]
                    geometry_name = mesh.graph.get(node_name)[1]
                    geometry = mesh.geometry[geometry_name]
                    
                    # Apply the transform to the geometry
                    if isinstance(geometry, trimesh.Trimesh):
                        transformed_mesh = geometry.copy()
                        transformed_mesh.apply_transform(transform)
                        mesh_list.append(transformed_mesh)
                
                if len(mesh_list) > 0:
                    return mesh_list
                else:
                    logger.warning(f"Scene in {file_path} contains no geometries")
                    return None
            if isinstance(mesh, trimesh.Trimesh):
                mesh_list = [mesh]
                return mesh_list
        except Exception as e:
            logger.error(f"Failed to load mesh from {file_path}: {e}")
            return None
        
    def visualize_point_cloud(
            self,
            vertices_pred,
            vertices_gt,
            basename: Optional[str] = None,
        ):
        # Convert to numpy for trimesh
        for i in range(vertices_pred.shape[0]):
            pred_points = vertices_pred[i].cpu().numpy()
            gt_points = vertices_gt[i].cpu().numpy()
            # Create point cloud objects
            pred_cloud = trimesh.PointCloud(pred_points, colors=[255, 0, 0, 100])  # Red for prediction
            gt_cloud = trimesh.PointCloud(gt_points, colors=[0, 0, 255, 100])      # Blue for ground truth
            # Create a visualization scene
            scene = trimesh.Scene()
            scene.add_geometry(pred_cloud)
            scene.add_geometry(gt_cloud)
            # Create output directory if it doesn't exist
            vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../visualization')
            os.makedirs(vis_dir, exist_ok=True)
            timestamp = int(time.time())
            output_path = os.path.join(vis_dir, f'point_cloud_comparison_{timestamp}.ply')
            if basename is not None:
                output_path = os.path.join(vis_dir, f'{basename}_{i}_{timestamp}.ply')
            combined_points = np.vstack([pred_points, gt_points])
            pred_colors = np.tile([255, 0, 0, 255], (len(pred_points), 1)).astype(np.uint8)
            gt_colors = np.tile([0, 0, 255, 255], (len(gt_points), 1)).astype(np.uint8)
            combined_colors = np.vstack([pred_colors, gt_colors])
            combined_cloud = trimesh.PointCloud(combined_points, colors=combined_colors)
            combined_cloud.export(output_path)
    
    def test_step(self, pred_path, gt_path, render_cfg: Dict = None):
        try:
            pred_mesh_list = self.load_mesh(pred_path)
            gt_mesh_list = self.load_mesh(gt_path)
            
            if pred_mesh_list is None or gt_mesh_list is None:
                logger.error('model load failed, please check the input file format')
                return None
            
            basename = os.path.basename(pred_path).split(".")[0]
            metrics = self.compute_metrics(pred_mesh_list, gt_mesh_list, basename, pred_path=pred_path, render_cfg=render_cfg)
            
            for k, v in metrics.items():
                if k not in self.metrics_values:
                    self.metrics_values[k] = []
                self.metrics_values[k] = self.metrics_values[k] + v.tolist()
        
        except Exception as e:
            tqdm.write(f"Error processing {pred_path}: {e}")
            return None
        
        return metrics
    
    def test_directory(self):
        input_dir = Path(self.cfg.input_dir)
        gt_dir = Path(self.cfg.output_dir)
        if os.path.exists(os.path.join(self.get_save_dir(), self.cfg.filename + ".json")):
            os.system(f"rm -rf {os.path.join(self.get_save_dir(), self.cfg.filename + '.json')}")
        
        if not input_dir.exists() or not gt_dir.exists():
            logger.error(f'The input directory {input_dir} or ground truth directory {gt_dir} does not exist.')
            return
        
        pred_files = list(input_dir.glob(f"*.{self.cfg.input_format}"))
        pred_files.sort()
        # pred_files = pred_files[:100]

        total = len(pred_files)
        logger.info(f"Finded {total} prediction files in {input_dir}.")
        if self.cfg.gpu_num > 1:
            pred_files = pred_files[self.cfg.gpu_id::self.cfg.gpu_num]
        
        for pred_file in tqdm(pred_files, desc=f"Evaluating models on GPU {self.cfg.gpu_id}", position=self.cfg.gpu_id, leave=True, dynamic_ncols=True, unit="file", unit_scale=True):
            file_name = pred_file.stem
            gt_file = gt_dir / f"{file_name}.{self.cfg.gt_format}"
            
            if not gt_file.exists():
                logger.warning(f"Ground truth file not found for {pred_file.name}, skipping.")
                continue
            
            render_cfg = None
            if self.cfg.render_eval:
                render_cfg = self.scene_render_config.get(file_name, None)
                if not render_cfg:
                    logger.warning(f"No render configuration found for {file_name}, skipping rendering.")
            
            self.test_step(str(pred_file), str(gt_file), render_cfg=render_cfg)

            torch.cuda.empty_cache()
                
    def on_test_end(self):
        if self.cfg.gpu_num == 1:
            final_metrics = {}
            for k, values in self.metrics_values.items():
                if values:
                    final_metrics[k] = sum(values) / len(values)

            if self.cfg.render_eval:
                self.fid_pred_images = list(set(self.fid_pred_images))
                self.fid_gt_render_images = list(set(self.fid_gt_render_images))
                self.fid_gt_scene_images = list(set(self.fid_gt_scene_images))

                self.fid_pred_images = [np.array(Image.open(img).convert("RGB")) for img in self.fid_pred_images]
                self.fid_gt_render_images = [np.array(Image.open(img).convert("RGB")) for img in self.fid_gt_render_images]
                self.fid_gt_scene_images = [np.array(Image.open(img).convert("RGB")) for img in self.fid_gt_scene_images]

                fid_render = compute_fid(self.fid_gt_render_images, self.fid_pred_images)
                final_metrics["render_fid"] = fid_render

                fid_scene = compute_fid(self.fid_gt_scene_images, self.fid_pred_images)
                final_metrics["scene_fid"] = fid_scene

            self.save_metrics_to_csv(final_metrics, self.cfg.filename)
            
            logger.info("The final evaluation metrics are:")
            sorted_metrics = sorted(final_metrics.items(), key=lambda item: item[0].lower())
            for k, v in sorted_metrics:
                logger.info(f"{k}: {v}")
            
            super().on_test_end() 
        else:
            import json
            if os.path.exists(os.path.join(self.get_save_dir(), self.cfg.filename + ".json")):
                with open(os.path.join(self.get_save_dir(), self.cfg.filename + ".json"), 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {}
            if self.cfg.render_eval:
                self.metrics_values["pred_render_images"] = self.fid_pred_images
                self.metrics_values["gt_render_images"] = self.fid_gt_render_images
                self.metrics_values["gt_scene_images"] = self.fid_gt_scene_images
            for k, values in self.metrics_values.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k] += values
            with open(os.path.join(self.get_save_dir(), self.cfg.filename + ".json"), 'w') as f:
                json.dump(metrics, f, indent=4)


            self.fid_pred_images = list(set(metrics.get("pred_render_images", [])))
            self.fid_gt_render_images = list(set(metrics.get("gt_render_images", [])))
            self.fid_gt_scene_images = list(set(metrics.get("gt_scene_images", [])))

            def _load_rgb_np(img_path: str) -> np.ndarray:
                with Image.open(img_path) as img:
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background
                    else:
                        img = img.convert('RGB')
                    return np.array(img)

            self.fid_pred_images = [_load_rgb_np(img) for img in self.fid_pred_images]
            self.fid_gt_render_images = [_load_rgb_np(img) for img in self.fid_gt_render_images]
            self.fid_gt_scene_images = [_load_rgb_np(img) for img in self.fid_gt_scene_images]

            if self.cfg.render_eval:
                fid_render = compute_fid(self.fid_gt_render_images, self.fid_pred_images)
                metrics["render_fid"] = [fid_render]

                fid_scene = compute_fid(self.fid_gt_scene_images, self.fid_pred_images)
                metrics["scene_fid"] = [fid_scene]
            
            metrics.pop("pred_render_images", None)
            metrics.pop("gt_render_images", None)
            metrics.pop("gt_scene_images", None)
            
            final_metrics = {}
            for k, values in metrics.items():
                if values:
                    final_metrics[k] = sum(values) / len(values)
            
            self.save_metrics_to_csv(final_metrics, self.cfg.filename)
            logger.info("The final evaluation metrics are:")
            sorted_metrics = sorted(final_metrics.items(), key=lambda item: item[0].lower())
            for k, v in sorted_metrics:
                logger.info(f"{k}: {v}")
            
            super().on_test_end()