import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin, StandardDatasetBaseVGGT, ImageConditionedVGGTMixin
from .. import models
from ..utils.dist_utils import read_file_dist
from scipy.spatial.transform import Rotation as R


class SparseStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        pose_encoding_type: str = "absT_quatR_S",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        self.pose_encoding_type = pose_encoding_type
        
    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(read_file_dist(ckpt_path), map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
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
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Build each representation
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords.float() / resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)
    
    def visualize_scene(self, x_0:Union[torch.Tensor, dict], positions:Union[torch.Tensor, dict] = None):
        positions = positions if isinstance(positions, torch.Tensor) else x_0['positions']
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 1024
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
            
            # # Rotate camera 90 degrees around its z-axis (camera's viewing direction)
            # # Create rotation matrix for 90 degree rotation around z-axis
            # rot_z_90 = torch.tensor([
            #     [0, -1, 0, 0],
            #     [1, 0, 0, 0],
            #     [0, 0, 1, 0],
            #     [0, 0, 0, 1]
            # ], dtype=torch.float).cuda()
            
            # # Apply rotation to extrinsics
            # extrinsics = torch.matmul(extrinsics, rot_z_90)
            
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)
        
        images = []
        x_0 = x_0.cuda()
        positions = positions.cuda()
        scene_positions = []
        representation = Octree(
            depth=10,
            aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
            device='cuda',
            primitive='voxel',
            sh_degree=0,
            primitive_config={'solid': True},
        )

        # def transform_from_query_frame(local_pos, local_euler, ref_pos, ref_euler):
        #     if local_euler.shape[0] == 4:
        #         rot_ref = R.from_quat(ref_euler, scalar_first=True).as_matrix()
        #         local_rot = R.from_quat(local_euler, scalar_first=True).as_matrix()
        #     else:
        #         # Convert euler angles to rotation matrices
        #         rot_ref = R.from_euler('xyz', ref_euler, degrees=True).as_matrix()
        #         local_rot = R.from_euler('xyz', local_euler, degrees=True).as_matrix()
            
        #     # Compute global position
        #     global_pos = np.dot(rot_ref, local_pos) + np.array(ref_pos)
            
        #     # Compute global rotation
        #     global_rot = np.dot(rot_ref, local_rot)
        #     if local_euler.shape[0] == 4:
        #         global_euler = R.from_matrix(global_rot).as_quat(scalar_first=True)
        #     else:
        #         global_euler = R.from_matrix(global_rot).as_euler('xyz', degrees=True)
            
        #     return global_pos, global_euler

        # if self.pose_encoding_type == "absT_eulerR_S":
        #     query_translation = positions[0, 0:3].float().cpu().numpy()
        #     query_rotation = positions[0, 3:6].float().cpu().numpy()
        #     query_rotation[2] = 90
        #     positions[0, 3:6] = torch.tensor(query_rotation).float().cuda()

        #     for i in range(1, x_0.shape[0]):
        #         object_translation = positions[i, 0:3].float().cpu().numpy()
        #         object_rotation = positions[i, 3:6].float().cpu().numpy()

        #         object_translation, object_rotation = transform_from_query_frame(
        #             object_translation, object_rotation, query_translation, query_rotation
        #         )
        #         positions[i, 0:3] = torch.tensor(object_translation).float().cuda()
        #         positions[i, 3:6] = torch.tensor(object_rotation).float().cuda()
        # elif self.pose_encoding_type == "absT_quatR_S":
        #     query_translation = positions[0, 0:3].float().cpu().numpy()
        #     positions[0, 3:7] = torch.tensor(R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(scalar_first=True)).float().cuda()
        #     query_rotation = positions[0, 3:7].float().cpu().numpy()

        #     for i in range(1, x_0.shape[0]):
        #         object_translation = positions[i, 0:3].float().cpu().numpy()
        #         object_rotation = positions[i, 3:7].float().cpu().numpy()

        #         object_translation, object_rotation = transform_from_query_frame(
        #             object_translation, object_rotation, query_translation, query_rotation
        #         )
        #         positions[i, 0:3] = torch.tensor(object_translation).float().cuda()
        #         positions[i, 3:7] = torch.tensor(object_rotation).float().cuda()

        for i in range(x_0.shape[0]):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            org_position = coords.float() / resolution
            translation = positions[i, 0:3].float()
            if self.pose_encoding_type == "absT_eulerR_S":
                rotation = positions[i, 3:6].float()
                scale = positions[i, 6].float()

                centered_position = org_position - 0.5
                scaled_position = centered_position * scale 

                # Convert Euler angles from PyTorch to numpy
                euler_angles = rotation.cpu().numpy()  # [roll, pitch, yaw]

                # Create rotation matrix using scipy (xyz order matches the original implementation)
                rot_matrix_np = R.from_euler('xyz', [euler_angles], degrees=True).as_matrix()[0]
            elif self.pose_encoding_type == "absT_quatR_S":
                rotation = positions[i, 3:7].float()
                scale = positions[i, 7].float()

                centered_position = org_position - 0.5
                scaled_position = centered_position * scale 

                quat_angles = rotation.cpu().numpy()  # [w, x, y, z]
                # Convert quaternion to rotation matrix
                try:
                    rot_matrix_np = R.from_quat(quat_angles, scalar_first=True).as_matrix()
                except ValueError as e:
                    print(f"Error: {e}. Quaternion: {quat_angles}")
                    quat_angles = [1, 0, 0, 0]
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
        if scene_positions.numel() != 0:
            x_max, x_min = scene_positions[:, 0].max(), scene_positions[:, 0].min()
            y_max, y_min = scene_positions[:, 1].max(), scene_positions[:, 1].min()
            z_max, z_min = scene_positions[:, 2].max(), scene_positions[:, 2].min()
        else:
            x_max, x_min = -1, 1
            y_max, y_min = -1, 1
            z_max, z_min = -1, 1

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
        images.append(image)
        return torch.stack(images)

class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'ss_latent_{self.latent_model}']]
        stats['With sparse structure latents'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance):
        latent = np.load(os.path.join(root, 'ss_latents', self.latent_model, f'{instance}.npz'))
        z = torch.tensor(latent['mean']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
        }
        return pack
    

class TextConditionedSparseStructureLatent(TextConditionedMixin, SparseStructureLatent):
    """
    Text-conditioned sparse structure dataset
    """
    pass


class ImageConditionedSparseStructureLatent(ImageConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """
    pass


class SparseStructureLatentVGGT(SparseStructureLatentVisMixin, StandardDatasetBaseVGGT):
    """
    Sparse structure latent dataset for VGGT
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
            **kwargs
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'ss_latent_{self.latent_model}']]
        stats['With sparse structure latents'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['vggt_feature'].notna()]
        stats['With VGGT features'] = len(metadata)
        return metadata, stats

    def get_instance(
        self,
        root: str,
        scene_id: str,
        instances: List[str],
        image_paths: List[str],
        scene_image_path: str,
    ):
        pack = {}
        latents = []
        for instance in instances:
            latent = np.load(os.path.join(root, 'ss_latents', self.latent_model, f'{instance}.npz'))
            z = torch.tensor(latent['mean']).float()
            if self.normalization is not None:
                z = (z - self.mean) / self.std
            latents.append(z)
        pack['x_0'] = torch.stack(latents, dim=0)
        return pack

class ImageConditionedSparseStructureLatentVGGT(ImageConditionedVGGTMixin, SparseStructureLatentVGGT):
    """
    Image-conditioned sparse structure dataset for VGGT
    """
    pass