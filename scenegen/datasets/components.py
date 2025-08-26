from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import numpy as np

class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])
        pack['cond'] = text
        return pack
    
    
class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack
    

class StandardDatasetBaseVGGT(Dataset):
    """
    Base class for stantard scene datasets with VGGT features.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(
        self,
        roots: str,
        max_batch_size: int = 4,
        max_gpu_batch_size: int = 4,
        use_single_asset: bool = True,
        data_file_name: str = 'metadata_scene',
        pose_encoding_type: str = "absT_quatR_S",
        use_mask_cond: bool = True,
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        self._stats = {}
        self.pose_encoding_type = pose_encoding_type
        self.use_mask_cond = use_mask_cond

        self.max_batch_size = max_batch_size
        self.max_gpu_batch_size = max_gpu_batch_size
        self.use_single_asset = use_single_asset

        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata_scene = pd.read_csv(os.path.join(root, f'{data_file_name}.csv'), low_memory=False, dtype={'scene_id': str})
            self._stats[key]['Total'] = len(metadata_scene)
            metadata_scene, stats = self.filter_metadata(metadata_scene)
            self._stats[key].update(stats)
            metadata_scene, stats = self.filter_metadata_scene(metadata_scene)
            self._stats[key].update(stats)
            metadata_scene['root'] = root
            metadata_scene['dataset_name'] = key
            self.metadata = pd.concat([self.metadata, metadata_scene])
            
            for bs in range(1, self.max_batch_size + 1):
                scene_occurrences = metadata_scene['scene_id'].value_counts()
                scene_occurrences = scene_occurrences[scene_occurrences == bs]
                scene_occurrences = scene_occurrences.index
                self._stats[key][f'Scene in dataset {key} with {bs} instances'] = len(scene_occurrences)
                for scene_id in scene_occurrences:
                    scene_metadata = metadata_scene[metadata_scene['scene_id'] == scene_id]
                    assert len(scene_metadata) == bs
                    for query_asset_idx in range(bs):
                        self.instances.append((root, scene_id, query_asset_idx, bs))

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, scene_id, query_asset_idx, bs = self.instances[index]
            instances_info = self.metadata.loc[self.metadata['scene_id'] == scene_id]
            asset_sh256s = instances_info['sha256'].values.tolist()
            
            # Create indices for non-query assets and randomly shuffle them
            non_query_indices = list(range(bs))
            non_query_indices.pop(query_asset_idx)
            np.random.shuffle(non_query_indices)
            non_query_indices = non_query_indices[:self.max_gpu_batch_size - 1]
            
            # Create ordered indices with query first, followed by randomly shuffled others
            ordered_indices = [query_asset_idx] + non_query_indices
            
            # Reorder assets based on the new indices
            instances = [asset_sh256s[i] for i in ordered_indices]
            image_paths = [instances_info['masked_image_path'].values.tolist()[i] for i in ordered_indices]
            scene_image_path = instances_info['scene_image_path'].values.tolist()[0]

            if bs > 1:
                translations = [json.loads(instances_info['translation'].values.tolist()[i]) for i in ordered_indices]
                
                scales = [float(instances_info['scale'].values.tolist()[i]) for i in ordered_indices]
                translations = [[translation[0] / scales[0], translation[1] / scales[0], translation[2] / scales[0]] for translation in translations]
                
                euler_rotations = [json.loads(instances_info['rotation'].values.tolist()[i]) for i in ordered_indices]
                
                # positions = [[translation, euler, scale] * bs] of shape (bs, 7) or [[translation, quat, scale] * bs] of shape (bs, 8)
                # positions = [[0, 0, 0, 0, 0, 0, 1]] or [[0, 0, 0, 1, 0, 0, 0, 1]]
                positions = []
                for i in range(len(ordered_indices)):
                    translation = translations[i]
                    euler_rotation = euler_rotations[i]
                    scale = scales[i] / scales[0]
                    relative_pos, relative_euler = self.transform_to_query_frame(
                        translation,
                        euler_rotation,
                        translations[0],
                        euler_rotations[0],
                        pose_encoding_type=self.pose_encoding_type,
                    )
                    if self.pose_encoding_type == "absT_eulerR_S":
                        positions.append([
                            relative_pos[0],
                            relative_pos[1],
                            relative_pos[2],
                            relative_euler[0],
                            relative_euler[1],
                            relative_euler[2],
                            scale,
                            ])
                    elif self.pose_encoding_type == "absT_quatR_S":
                        positions.append([
                            relative_pos[0],
                            relative_pos[1],
                            relative_pos[2],
                            relative_euler[0],
                            relative_euler[1],
                            relative_euler[2],
                            relative_euler[3],
                            scale,
                            ])
                positions = torch.tensor(positions).float()
            else:
                if self.pose_encoding_type == "absT_eulerR_S":
                    positions = [[0, 0, 0, 0, 0, 0, 1]]
                elif self.pose_encoding_type == "absT_quatR_S":
                    positions = [[0, 0, 0, 1, 0, 0, 0, 1]]
                positions = torch.tensor(positions).float()

            pack = self.get_instance(
                root,
                scene_id,
                instances,
                image_paths,
                scene_image_path,
                )
            pack['positions'] = positions
            return pack
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
    
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total training case: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)

    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        return metadata, {}

    def filter_metadata_scene(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        stats = {}
        scene_id_series = metadata['scene_id'].astype(str)
        mask_scene = scene_id_series.str.len() == 7
        scene_occurrences = metadata.loc[mask_scene, 'scene_id'].value_counts()

        # Filter out scene_ids that appear more than max_batch_size times
        frequent_scene_ids = scene_occurrences[scene_occurrences > self.max_batch_size].index
        mask_frequent = ~metadata['scene_id'].isin(frequent_scene_ids)
        if self.use_single_asset:
            filtered_metadata = metadata[mask_frequent]
        else:
            filtered_metadata = metadata[mask_scene & mask_frequent]
        metadata = filtered_metadata
        stats['Scene with instance < {}'.format(self.max_batch_size)] = len(metadata)

        return metadata, stats

    def transform_to_query_frame(self, pos_target, euler_target, pos_query, euler_query, pose_encoding_type="absT_quatR_S"):

        rot_query = R.from_euler('xyz', euler_query, degrees=True).as_matrix()
        rot_target = R.from_euler('xyz', euler_target, degrees=True).as_matrix()

        relative_pos = np.dot(rot_query.T, np.array(pos_target) - np.array(pos_query))

        relative_rot = np.dot(rot_query.T, rot_target)
        if pose_encoding_type == "absT_eulerR_S":
            relative_euler = R.from_matrix(relative_rot).as_euler('xyz', degrees=True)
        elif pose_encoding_type == "absT_quatR_S":
            relative_euler = R.from_matrix(relative_rot).as_quat(scalar_first=True)

        return relative_pos, relative_euler
        
    @abstractmethod
    def get_instance(
        self,
        root: str,
        scene_id: str,
        instances: List[str],
        image_paths: List[str],
        scene_image_path: str,
    ) -> Dict[str, Any]:
        pass

class ImageConditionedVGGTMixin:
    def __init__(self, roots, *, image_size=518, use_mask_cond=True, **kwargs):
        self.use_mask_cond = use_mask_cond
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        metadata = metadata[metadata['masked_image_path'].notna()]
        stats['Masked image'] = len(metadata)
        metadata = metadata[metadata['scene_image_path'].notna()]
        stats['Scene image'] = len(metadata)
        return metadata, stats

    def get_instance(
        self,
        root: str,
        scene_id: str,
        instances: List[str],
        image_paths: List[str],
        scene_image_path: str,
    ):
        pack = super().get_instance(root, scene_id, instances, image_paths, scene_image_path)
        
        image_paths = [os.path.join(root, image_path) for image_path in image_paths]
        images = [Image.open(image_path) for image_path in image_paths]
        mask_images = []  
        
        alphas = [np.array(image.getchannel(3)) for image in images]
        bboxes = [np.array(alpha).nonzero() for alpha in alphas]
        bboxes = [[bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()] for bbox in bboxes]
        centers = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in bboxes]
        hsizes = [max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 for bbox in bboxes]
        aug_size_ratio = 1.2
        aug_hsizes = [hsize * aug_size_ratio for hsize in hsizes]
        aug_center_offsets = [[0, 0] for _ in range(len(images))]
        aug_centers = [[center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]] for center, aug_center_offset in zip(centers, aug_center_offsets)]
        aug_bboxes = [[int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)] for aug_center, aug_hsize in zip(aug_centers, aug_hsizes)]
        images = [image.crop(aug_bbox) for image, aug_bbox in zip(images, aug_bboxes)]
        images = [image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS) for image in images]

        if self.use_mask_cond:
            mask_images_paths = [''.join(image_path.split('.')[:-1]) + '_mask.png' for image_path in image_paths]
            if all(os.path.exists(mask_image_path) for mask_image_path in mask_images_paths):
                mask_images = [Image.open(mask_image_path) for mask_image_path in mask_images_paths]
                mask_images = [mask_image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS) for mask_image in mask_images]
                mask_images = [torch.tensor(np.array(mask_image)).permute(2, 0, 1).float() / 255.0 for mask_image in mask_images]
            else:
                single_image = [Image.open(image_path) for image_path in image_paths]
                single_image = [image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS) for image in single_image]
                mask_images = [np.array(single_image.getchannel(3)) for _ in range(len(images))]
                mask_images = [torch.tensor(np.array(mask_image)).repeat(3, 1, 1).permute(2, 0, 1).float() / 255.0 for mask_image in mask_images]

        alphas = [image.getchannel(3) for image in images]
        images = [image.convert('RGB') for image in images]
        images = [torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0 for image in images]
        alphas = [torch.tensor(np.array(alpha)).float() / 255.0 for alpha in alphas]
        images = [image * alpha.unsqueeze(0) for image, alpha in zip(images, alphas)]

        scene_image_path = os.path.join(root, scene_image_path)
        scene_image = Image.open(scene_image_path)
        scene_image = scene_image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        scene_image = scene_image.convert('RGB')
        scene_image = torch.tensor(np.array(scene_image)).permute(2, 0, 1).float() / 255.0

        vggt_feature = np.load(os.path.join(root, 'vggt_features', scene_id + '.npz'))
        if 'arr_0' in vggt_feature:
            vggt_feature = torch.tensor(vggt_feature['arr_0']).float()
        else:
            key = list(vggt_feature.keys())[0]
            vggt_feature = torch.tensor(vggt_feature[key]).float()

        pack['cond'] = torch.stack(images, dim=0)
        pack['scene_image'] = scene_image
        if self.use_mask_cond:
            pack['mask_cond'] = torch.stack(mask_images, dim=0)
        pack['vggt_feature'] = vggt_feature

        return pack