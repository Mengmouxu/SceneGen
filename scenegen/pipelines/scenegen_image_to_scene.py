from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vggt'))
from vggt.models.vggt import VGGT
import utils3d
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from scipy.spatial.transform import Rotation as R
from ..utils import postprocessing_utils
import trimesh
import io
from contextlib import redirect_stdout

class SceneGenImageToScenePipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "SceneGenImageToScenePipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(SceneGenImageToScenePipeline, SceneGenImageToScenePipeline).from_pretrained(path)
        new_pipeline = SceneGenImageToScenePipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        print(f"Loading image conditioning model: {args['image_cond_model']}")
        new_pipeline._init_image_cond_model(args['image_cond_model'])
        print(f"Loading vggt model: {args['vggt_model']}")
        new_pipeline._init_vggt_model(args['vggt_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """

        cache_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        
        if os.path.exists(cache_dir):
            dinov2_model = torch.hub.load(cache_dir, name, source='local', pretrained=True)
        else:
            dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
    
    def _init_vggt_model(self, name: str):
        """
        Initialize the vggt feature extractor.
        """
        vggt_model = VGGT.from_pretrained(name).to(self.device)
        vggt_model.eval()
        self.models['vggt_model'] = vggt_model

    def preprocess_vggt_image(self, input: Image.Image) -> Image.Image:
        target_size = 518
        if input.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", input.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            input = Image.alpha_composite(background, input)
        input = input.convert("RGB")
        width, height = input.size
        if width > height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        
        input = input.resize((new_width, new_height), Image.Resampling.BICUBIC)
        input = transforms.ToTensor()(input)

        h_padding = target_size - input.shape[1]
        w_padding = target_size - input.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=1.0)
        
        input = input.unsqueeze(0).unsqueeze(0)
        input = input.to(self.device)
        return input

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    @torch.no_grad()
    def encode_vggt_image(self, image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Encode the image using the vggt model.

        Args:
            image (Union[torch.Tensor, Image.Image]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, Image.Image):
            image = self.preprocess_vggt_image(image)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim == 4:
                assert image.shape[0] == 1, "Image tensor should be single image (1, C, H, W)"
            image = image.to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        prediction, _ = self.models['vggt_model'].aggregator(image)
        features = prediction[-1][0, ...]
        return features
    
    @torch.no_grad()
    def encode_vggt_multi_image(self, image: List[Image.Image]) -> torch.Tensor:
        
        for i in range(len(image)):
            image[i] = self.preprocess_vggt_image(image[i])
        image = torch.cat(image, dim=0).to(self.device)
        
        prediction, _ = self.models['vggt_model'].aggregator(image)
        features = prediction[-1]
        return features
    
    @torch.no_grad()
    def get_cond(
        self,
        image: Union[torch.Tensor, list[Image.Image]],
        scene_image: Union[torch.Tensor, Image.Image],
        mask_image: Union[torch.Tensor, list[Image.Image]] = None,
        use_mask: bool = True,
        use_scene_image_cond: bool = True,
        use_vggt_feature_cond: bool = True,
    ) -> dict:
        """
        Get the conditioning information for the model with optional mask, scene image,
        and VGGT feature conditions controlled by flags. If a condition is not used,
        it is skipped (no zero placeholders).
        """
        # Base image condition
        cond = self.encode_image(image)

        # Optional mask condition
        if use_mask and mask_image is not None:
            if isinstance(mask_image, torch.Tensor) and mask_image.ndim == 3:
                mask_image = mask_image.unsqueeze(0)
            mask_cond = self.encode_image(mask_image)
            cond = torch.cat([cond, mask_cond], dim=1)

        # Optional scene image condition
        if use_scene_image_cond:
            if isinstance(scene_image, torch.Tensor):
                si = scene_image
                if si.ndim == 3:
                    si = si.unsqueeze(0)
                assert si.ndim == 4 and si.shape[0] == 1, "scene_image tensor must be shaped (1, C, H, W)"
                scene_cond = self.encode_image(si)
            else:
                scene_cond = self.encode_image([scene_image])

            scene_cond = scene_cond.expand(cond.shape[0], -1, -1)
            cond = torch.cat([cond, scene_cond], dim=1)

        # Optional VGGT feature condition
        if use_vggt_feature_cond:
            scene_vggt_feature = self.encode_vggt_image(scene_image)
            if scene_vggt_feature.ndim == 2:
                scene_vggt_feature = scene_vggt_feature.unsqueeze(0)  # (1, T, C)
            # Split channels and concat along token dim to match training behavior
            scene_vggt_feature = torch.cat(
                [scene_vggt_feature[..., :1024], scene_vggt_feature[..., 1024:]],
                dim=1,
            )
            scene_vggt_feature = scene_vggt_feature.expand(cond.shape[0], -1, -1)
            cond = torch.cat([cond, scene_vggt_feature], dim=1)

        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    @torch.no_grad()
    def get_multi_cond(
            self,
            image: List[List[Image.Image]],
            scene_image: List[Union[torch.Tensor, Image.Image]],
            mask_image: List[List[Image.Image]] = None,
        ) -> list[dict]:
        scene_cond = self.encode_image(scene_image)
        scene_vggt_feature = self.encode_vggt_multi_image(scene_image)
        multi_cond = []
        for i in range(len(image)):
            cond = self.encode_image(image[i])
            if mask_image is not None:
                mask_cond = self.encode_image(mask_image[i])
                cond = torch.cat([cond, mask_cond], dim=1)

            scene_vggt_feature_i = torch.cat([scene_vggt_feature[i, ..., :1024], scene_vggt_feature[i, ..., 1024:]], dim=1)

            scene_cond_i = scene_cond[i].expand(cond.shape[0], -1, -1)
            scene_vggt_feature_i = scene_vggt_feature_i.expand(cond.shape[0], -1, -1)
            cond = torch.cat([cond, scene_cond_i, scene_vggt_feature_i], dim=1)
            neg_cond = torch.zeros_like(cond)
            multi_cond.append({
                'cond': cond,
                'neg_cond': neg_cond,
            })
        return multi_cond
    
    @torch.no_grad()
    def get_cond_org(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    @torch.no_grad()
    def get_multi_cond_org(self, image: List[List[Image.Image]]) -> list[dict]:
        multi_cond = []
        for i in range(len(image)):
            cond = self.encode_image(image[i])
            neg_cond = torch.zeros_like(cond)
            multi_cond.append({
                'cond': cond,
                'neg_cond': neg_cond,
            })
        return multi_cond
    
    @torch.no_grad()
    def sample_sparse_structure(
        self,
        cond: dict,
        sampler_params: dict = {},
        get_voxel_vis: bool = True,
        positions_type: str = 'last',
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            sampler_params (dict): Additional parameters for the sampler.
            get_voxel_vis (bool): Whether to generate voxel visualization.
            positions_type (str): The type of positions to use ('avg' or 'last').
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        if cond['cond'].ndim == 3:
            batch_size = cond['cond'].shape[0]
        elif cond['cond'].ndim == 4:
            batch_size = cond['cond'].shape[1]
        
        noise = torch.randn(batch_size, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        ret = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )
        z_s = ret.samples
        
        if positions_type == 'avg':
            positions = ret.pred_pos_t
            if cond['cond'].ndim == 3:
                positions = ret.positions
            elif cond['cond'].ndim == 4:
                positions = positions[-cond['cond'].shape[0]:]
                positions = torch.stack(positions, dim=0)
                positions = torch.mean(positions, dim=0, keepdim=True).squeeze(0)
        elif positions_type == 'last':
            positions = ret.positions
        else:
            raise ValueError(f"Unsupported positions type: {positions_type}")
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        if get_voxel_vis:
            renderer = OctreeRenderer()
            renderer.rendering_options.resolution = 512
            renderer.rendering_options.near = 0.8
            renderer.rendering_options.far = 1.6
            renderer.rendering_options.bg_color = (0, 0, 0)
            renderer.rendering_options.ssaa = 4
            renderer.pipe.primitive = 'voxel'

            yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
            yaws = [y + yaws_offset for y in yaws]
            pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

            exts = []
            ints = []
            for yaw, pitch in zip(yaws, pitch):
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(pitch),
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(pitch),
                ]).float().cuda() * 2
                fov = torch.deg2rad(torch.tensor(40)).cuda()
                extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda()) 
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                exts.append(extrinsics)
                ints.append(intrinsics)
            
            images = []
            x_0 = decoder(z_s)
            for i in range(batch_size):
                representation = Octree(
                    depth=10,
                    aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                    device='cuda',
                    primitive='voxel',
                    sh_degree=0,
                    primitive_config={'solid': True},
                )
                coords_vis = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
                resolution = x_0.shape[-1]
                representation.position = coords_vis.float() / resolution
                representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

                image = torch.zeros(3, 1024, 1024).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
                images.append(image)

            scene_positions = []
            renderer = OctreeRenderer()
            renderer.rendering_options.resolution = 1024
            renderer.rendering_options.near = 0.8
            renderer.rendering_options.far = 1.6
            renderer.rendering_options.bg_color = (0, 0, 0)
            renderer.rendering_options.ssaa = 4
            renderer.pipe.primitive = 'voxel'
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )

            for i in range(x_0.shape[0]):
                coords_i = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
                resolution = x_0.shape[-1]
                org_position = coords_i.float() / resolution
                translation = positions[i, 0:3].float()
                rotation = positions[i, 3:7].float()
                scale = positions[i, 7].float()

                centered_position = org_position - 0.5
                scaled_position = centered_position * scale 

                quat_angles = rotation.cpu().numpy()  # [w, x, y, z]
                # Convert quaternion to rotation matrix
                rot_matrix_np = R.from_quat(quat_angles, scalar_first=True).as_matrix()

                rot_x_np = R.from_euler("x",np.array([-90, 0, 0]), degrees=True).as_matrix()[0]
                rot_np = rot_x_np
                rot_matrix_np = np.dot(rot_matrix_np, rot_np)
                
                # Convert back to PyTorch tensor on the same device
                rot_matrix = torch.tensor(rot_matrix_np, device=rotation.device, dtype=rotation.dtype)
                
                rotated_position = torch.matmul(scaled_position, rot_matrix.T)

                final_position = rotated_position + 0.5 + translation
                scene_positions.append(final_position)
            scene_positions = torch.cat(scene_positions, dim=0)
            x_max, x_min = scene_positions[:, 0].max(), scene_positions[:, 0].min()
            y_max, y_min = scene_positions[:, 1].max(), scene_positions[:, 1].min()
            z_max, z_min = scene_positions[:, 2].max(), scene_positions[:, 2].min()

            edge_length = max(x_max - x_min, y_max - y_min, z_max - z_min)
            center = torch.tensor([
                (x_max + x_min) / 2, 
                (y_max + y_min) / 2, 
                (z_max + z_min) / 2
            ], device='cuda')

            scene_positions = (scene_positions - center) / edge_length + 0.5

            representation.position = scene_positions
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(512)), dtype=torch.uint8, device='cuda')
            # Add coordinate axes visualization
            axis_length = 0.3  # Length of each axis
            num_points_per_axis = 50  # Number of points per axis

            # Create coordinate axes
            axes_positions = []
            axes_colors = []

            # Origin point (centered in the scene)
            origin = torch.tensor([0.5, 0.5, 0.5], device='cuda')

            # Create axes: X (red), Y (green), Z (blue)
            for i, (axis_dir, color) in enumerate(zip(
                [torch.tensor([1.0, 0.0, 0.0], device='cuda'),  # X-axis
                torch.tensor([0.0, 1.0, 0.0], device='cuda'),  # Y-axis 
                torch.tensor([0.0, 0.0, 1.0], device='cuda')], # Z-axis
                [torch.tensor([1.0, 0.0, 0.0], device='cuda'),  # Red
                torch.tensor([0.0, 1.0, 0.0], device='cuda'),  # Green
                torch.tensor([0.0, 0.0, 1.0], device='cuda')]  # Blue
            )):
                # Create points along the axis
                line_points = torch.linspace(0, axis_length, num_points_per_axis, device='cuda')
                for t in line_points:
                    pos = origin + axis_dir * t
                    axes_positions.append(pos)
                    axes_colors.append(color)

            # Convert lists to tensors
            axes_positions = torch.stack(axes_positions)
            axes_colors = torch.stack(axes_colors)

            # Add axes points to the representation
            representation.position = torch.cat([representation.position, axes_positions], dim=0)
            representation.depth = torch.cat([
                representation.depth,
                torch.full((axes_positions.shape[0], 1), int(np.log2(512)), dtype=torch.uint8, device='cuda')
            ], dim=0)

            # Create color map for rendering
            all_colors = torch.zeros((representation.position.shape[0], 3), device='cuda')
            all_colors[:scene_positions.shape[0]] = scene_positions  # Original voxel positions as colors
            all_colors[scene_positions.shape[0]:] = axes_colors      # Axis colors

            # Store this color map for later use in rendering
            scene_positions = all_colors  # This will be used as colors_overwrite in rendering

            image = torch.zeros(3, 2048, 2048).cuda()
            tile = [2, 2]
        
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=scene_positions)
                image[:, 1024 * (j // tile[1]):1024 * (j // tile[1] + 1), 1024 * (j % tile[1]):1024 * (j % tile[1] + 1)] = res['color']
            scene_images = [image]
            scene_images = torch.stack(scene_images)
            images = torch.stack(images)
        else:
            images = None
            scene_images = None

        return coords, positions, images, scene_images
    
    @torch.no_grad()
    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    @torch.no_grad()
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: List[Image.Image],
        scene_image: Union[torch.Tensor, Image.Image],
        mask_image: List[Image.Image] = None,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        positions_type: Literal['last', 'avg'] = 'last',
        preprocess_image: bool = True,
        get_voxel_vis: bool = True,
        use_mask: bool = True,
        use_scene_image_cond: bool = True,
        use_vggt_feature_cond: bool = True,
    ):
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
            get_voxel_vis (bool): Whether to get voxel visualization.
            positions_type (Literal['last', 'avg']): The type of positions to return.
            use_mask (bool): Whether to use mask condition.
            use_scene_image_cond (bool): Whether to use scene image condition.
            use_vggt_feature_cond (bool): Whether to use VGGT feature condition.
        """
        batch_size = len(image)

        if preprocess_image:
            print("Preprocessing image...")
            for i in range(batch_size):
                image[i] = self.preprocess_image(image[i])

        cond = self.get_cond(
            image,
            scene_image,
            mask_image,
            use_mask=use_mask,
            use_scene_image_cond=use_scene_image_cond,
            use_vggt_feature_cond=use_vggt_feature_cond
        )
        print(cond['cond'].shape)
        torch.manual_seed(seed)
        coords, positions, image_vis, scene_vis = self.sample_sparse_structure(
            cond,
            sparse_structure_sampler_params,
            get_voxel_vis=get_voxel_vis,
            positions_type=positions_type,
        )

        del cond

        cond = self.get_cond_org(image)
        slat = self.sample_slat(
            cond,
            coords,
            slat_sampler_params
        )

        scene = []

        for i in range(batch_size):
            scene_i = self.decode_slat(slat[i], formats)
            scene.append(scene_i)

        return scene, positions, image_vis, scene_vis


    def transform_from_local(self, local_pos, local_euler, ref_pos, ref_euler):
        # Convert euler angles to rotation matrices
        if len(ref_euler) == 4:
            ref_euler = R.from_quat(ref_euler, scalar_first=True).as_euler('xyz', degrees=True)
        if len(local_euler) == 4:
            local_euler = R.from_quat(local_euler, scalar_first=True).as_euler('xyz', degrees=True)
            
        rot_ref = R.from_euler('xyz', ref_euler, degrees=True).as_matrix()
        local_rot = R.from_euler('xyz', local_euler, degrees=True).as_matrix()
        
        # Compute global position
        global_pos = np.dot(rot_ref, local_pos) + np.array(ref_pos)
        
        # Compute global rotation
        global_rot = np.dot(rot_ref, local_rot)
        global_euler = R.from_matrix(global_rot).as_quat(scalar_first=True)
        
        return tuple(global_pos), tuple(global_euler)

    def run_scene(
        self,
        image:Union[List[Image.Image], List[List[Image.Image]]],
        scene_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        mask_image: Union[List[Image.Image], List[List[Image.Image]]] = None,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        positions_type: Literal['last', 'avg'] = 'last',
        preprocess_image: bool = True,
        resorted_indices: List[int] = None,
        simplify: float = 0.95,
        texture_size: int = 1024,
        use_mask: bool = True,
        use_scene_image_cond: bool = True,
        use_vggt_feature_cond: bool = True,
    ) -> dict:
        """
        Run the pipeline for scene generation.

        Args:
            image (Image.Image): The image prompt.
            scene_image (Image.Image): The scene image.
            mask_image (List[Image.Image]): The mask image.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            positions_type (Literal['last', 'avg']): The type of positions to return.
            preprocess_image (bool): Whether to preprocess the image.
            resorted_indices (List[int]): The indices to resort the output meshes.
            simplify (float): The ratio of triangles to remove in the simplification process.
            texture_size (int): The size of the texture used for the GLB.
            use_mask (bool): Whether to use mask condition.
            use_scene_image_cond (bool): Whether to use scene image condition.
            use_vggt_feature_cond (bool): Whether to use VGGT feature condition.
        """
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            results = self.run(
                image,
                scene_image,
                mask_image,
                seed,
                sparse_structure_sampler_params,
                slat_sampler_params,
                formats=['mesh', 'gaussian'],
                preprocess_image=preprocess_image,
                positions_type=positions_type,
                get_voxel_vis= False,
                use_mask=use_mask,
                use_scene_image_cond=use_scene_image_cond,
                use_vggt_feature_cond=use_vggt_feature_cond,
            )
        elif isinstance(image, list) and isinstance(image[0], list):
            results = self.run_multi_image(
                image,
                scene_image,
                mask_image,
                seed,
                sparse_structure_sampler_params,
                slat_sampler_params,
                formats=['mesh', 'gaussian'],
                preprocess_image=preprocess_image,
                positions_type=positions_type,
                get_voxel_vis=False,
            )
        outputs = results[0]
        positions = results[1].cpu().numpy()

        pos_query = (0.0, 0.0, 0.0)
        quat_query = R.from_euler('xyz', (90, 0, 0), degrees=True).as_quat(scalar_first=True)

        local_pos, local_quat, local_scale = [], [], []

        for i in range(1, len(positions)):
            local_pos.append(np.array(positions[i][0:3]))
            local_quat.append(np.array(positions[i][3:7]))
            local_scale.append(float(positions[i][7]))

        for i in range(len(local_pos)):
            pos = self.transform_from_local(local_pos[i], local_quat[i], pos_query, quat_query)
            local_pos[i] = pos[0]
            local_quat[i] = pos[1]
        
        positions = np.array([pos_query] + local_pos)
        quats = np.array([quat_query] + local_quat)
        scales = np.array([1.0] + local_scale)

        trimeshes = []
        for i in range(len(outputs)):
            with redirect_stdout(io.StringIO()):  # Redirect stdout to a string buffer
                glb = postprocessing_utils.to_glb(
                    outputs[i]['gaussian'][0],
                    outputs[i]['mesh'][0],
                    # Optional parameters
                    simplify=simplify,          # Ratio of triangles to remove in the simplification process
                    texture_size=texture_size,      # Size of the texture used for the GLB
                )
            trimeshes.append(glb)

        # Compose the output meshes into a single scene
        scene = trimesh.Scene()
        # Add each mesh to the scene with the appropriate transformation
        if resorted_indices is None:
            resorted_indices = range(len(trimeshes))
        
        current_transform = np.eye(4)
        for i in resorted_indices:
            rmat = R.from_quat(quats[i], scalar_first=True).as_matrix() * scales[i]
            transform = np.eye(4)
            transform[:3, :3] = rmat
            transform[:3, 3] = positions[i]
            scene.add_geometry(trimeshes[i], transform=transform)
            if i == resorted_indices[0]:
                current_transform = transform

        # Move the query asset to the origin with no rotation
        R_matrix = current_transform[:3, :3]
        t = current_transform[:3, 3]
        T = np.eye(4)
        T[:3, :3] = R_matrix.T
        T[:3, 3] = -np.dot(R_matrix.T, t)
        scene.apply_transform(T)

        # Normalize the scene to fit within (-1, -1, -1) and (1, 1, 1) with a margin.
        bounds = scene.bounds
        scene_min, scene_max = bounds
        scene_center = (scene_min + scene_max) / 2.0
        extents = scene_max - scene_min
        max_extent = extents.max()

        # Define a margin (e.g., 2% margin from each side)
        margin = 0.02
        target_half_size = 1 - margin
        scale_factor = target_half_size * 2 / max_extent
        normalize_transform = trimesh.transformations.compose_matrix(
            translate=-scene_center,
            scale=[scale_factor, scale_factor, scale_factor]
        )

        scene.apply_transform(normalize_transform)

        positions = [positions[i] for i in resorted_indices]
        positions = np.array(positions)
        trimeshes = [trimeshes[i] for i in resorted_indices]

        outputs = {
            'scene': scene,
            'positions': positions,
            'assets': trimeshes,
        }

        return outputs

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                if len(cond_indices) > 1:
                    cond_idx = cond_indices.pop(0)
                    cond_i = cond[cond_idx]
                    return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
                if len(cond_indices) == 1:
                    pred_list = []
                    for i in range(len(cond)):
                        cond_i = cond[i]
                        pred_i = self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
                        pred_list.append(pred_i)
                    if isinstance(pred_list[0], tuple):
                        pred_avg = tuple(sum(p[i] for p in pred_list) / len(pred_list) for i in range(len(pred_list[0])))
                    else:
                        pred_avg = sum(pred_list) / len(pred_list)
                    return pred_avg
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')
    
    @torch.no_grad()
    def run_multi_image(
        self,
        image: List[List[Image.Image]],
        scene_image: List[Image.Image],
        mask_image: List[List[Image.Image]] = None,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
        positions_type: Literal['last', 'avg'] = 'last',
        preprocess_image: bool = True,
        get_voxel_vis: bool = True,
    ):
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
            get_voxel_vis (bool): Whether to get voxel visualization.
        """
        batch_size = len(image[0])
        view_num = len(image)

        if preprocess_image:
            print("Preprocessing image...")
            for i in range(view_num):
                for j in range(batch_size):
                    image[i][j] = self.preprocess_image(image[i][j])

        multi_cond = self.get_multi_cond(
            image,
            scene_image,
            mask_image,
        )
        cond = {
            'cond': torch.stack([c['cond'] for c in multi_cond], dim=0),
            'neg_cond': multi_cond[0]['neg_cond'],
        }
        torch.manual_seed(seed)

        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', view_num, ss_steps, mode=mode):
            coords, positions, image_vis, scene_vis = self.sample_sparse_structure(
                cond,
                sparse_structure_sampler_params,
                get_voxel_vis=get_voxel_vis,
                positions_type='last' if mode=='stochastic' else positions_type,
            )

        del cond, multi_cond

        multi_cond = self.get_multi_cond_org(image)
        cond = {
            'cond': torch.stack([c['cond'] for c in multi_cond], dim=0),
            'neg_cond': multi_cond[0]['neg_cond'],
        }

        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(image), slat_steps, mode=mode):
            slat = self.sample_slat(
                cond,
                coords,
                slat_sampler_params
            )

        scene = []

        for i in range(batch_size):
            scene_i = self.decode_slat(slat[i], formats)
            scene.append(scene_i)

        return scene, positions, image_vis, scene_vis