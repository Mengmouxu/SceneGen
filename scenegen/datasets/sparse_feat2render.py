import os
from PIL import Image
import json
import numpy as np
import pandas as pd
import torch
import utils3d.torch
from ..modules.sparse.basic import SparseTensor
from .components import StandardDatasetBase


class SparseFeat2Render(StandardDatasetBase):
    """
    SparseFeat2Render dataset.
    
    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        model (str): model name
        resolution (int): resolution of the data
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        model: str = 'dinov2_vitl14_reg',
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
    ):
        self.image_size = image_size
        self.model = model
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        
        super().__init__(roots)
        
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'feature_{self.model}']]
        stats['With features'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        return metadata, stats

    def _get_image(self, root, instance):
        with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]
        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        image_path = os.path.join(root, 'renders', instance, metadata['file_path'])
        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }
    
    def _get_feat(self, root, instance):
        DATA_RESOLUTION = 64
        feats_path = os.path.join(root, 'features', self.model, f'{instance}.npz')
        feats = np.load(feats_path, allow_pickle=True)
        coords = torch.tensor(feats['indices']).int()
        feats = torch.tensor(feats['patchtokens']).float()
        
        if self.resolution != DATA_RESOLUTION:
            factor = DATA_RESOLUTION // self.resolution
            coords = coords // factor
            coords, idx = coords.unique(return_inverse=True, dim=0)
            feats = torch.scatter_reduce(
                torch.zeros(coords.shape[0], feats.shape[1], device=feats.device),
                dim=0,
                index=idx.unsqueeze(-1).expand(-1, feats.shape[1]),
                src=feats,
                reduce='mean'
            )
        
        return {
            'coords': coords,
            'feats': feats,
        }

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return sample['image']

    @staticmethod
    def collate_fn(batch):
        pack = {}
        coords = []
        for i, b in enumerate(batch):
            coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
        coords = torch.cat(coords)
        feats = torch.cat([b['feats'] for b in batch])
        pack['feats'] = SparseTensor(
            coords=coords,
            feats=feats,
        )
        
        pack['image'] = torch.stack([b['image'] for b in batch])
        pack['alpha'] = torch.stack([b['alpha'] for b in batch])
        pack['extrinsics'] = torch.stack([b['extrinsics'] for b in batch])
        pack['intrinsics'] = torch.stack([b['intrinsics'] for b in batch])

        return pack

    def get_instance(self, root, instance):
        image = self._get_image(root, instance)
        feat = self._get_feat(root, instance)
        return {
            **image,
            **feat,
        }
