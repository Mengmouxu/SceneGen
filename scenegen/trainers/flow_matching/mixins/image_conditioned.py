from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import os

from ....utils import dist_utils


class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}

class ImageConditionedVGGTMixin:
    """
    Mixin for image-conditioned with scene image and VGGT features.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(
        self,
        *args,
        image_cond_model: str = 'dinov2_vitl14_reg',
        use_mask_cond: bool = True,
        use_scene_image_cond: bool = True,
        use_vggt_feature_cond: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        self.use_mask_cond = use_mask_cond
        self.use_scene_image_cond = use_scene_image_cond
        self.use_vggt_feature_cond = use_vggt_feature_cond
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        cache_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        
        if os.path.exists(cache_dir):
            torch.hub.load(cache_dir, image_cond_model, source='local', pretrained=True)
        else:
            torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        import warnings
        warnings.filterwarnings("ignore", message="xFormers is available")

        with dist_utils.local_master_first():
            cache_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
            if os.path.exists(cache_dir):
                dinov2_model = torch.hub.load(cache_dir, self.image_cond_model_name, source='local', pretrained=True)
            else:
                dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_image(cond)
        if "mask_cond" in kwargs:
            mask_cond = kwargs.pop("mask_cond", None)
            if self.use_mask_cond:
                mask_cond = self.encode_image(mask_cond)
                cond = torch.cat([cond, mask_cond], dim=1)

        assert "scene_image" in kwargs, "Scene image is not provided"
        scene_image = kwargs.pop("scene_image", None)
        if self.use_scene_image_cond and scene_image is not None:
            scene_image = scene_image.expand(cond.shape[0], *scene_image.shape[1:])
            scene_cond = self.encode_image(scene_image)
            cond = torch.cat([cond, scene_cond], dim=1)

        assert "vggt_feature" in kwargs, "VGGT feature is not provided"
        vggt_feature = kwargs.pop("vggt_feature", None)
        if self.use_vggt_feature_cond and vggt_feature is not None:
            vggt_feature = vggt_feature.expand(cond.shape[0], *vggt_feature.shape[1:])
            vggt_feature = torch.cat([vggt_feature[..., :1024], vggt_feature[..., 1024:]], dim=1)
            cond = torch.cat([cond, vggt_feature], dim=1)

        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        assert "scene_image" in kwargs, "Scene image is not provided"
        cond = torch.cat([cond, kwargs.pop("scene_image")], dim=0)
        return {'image': {'value': cond, 'type': 'image'}}
    
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data with scene image and VGGT features, controlled by flags.
        """
        cond = self.encode_image(cond)
        if self.use_mask_cond and "mask_cond" in kwargs:
            mask_cond = kwargs.pop("mask_cond")
            mask_cond = self.encode_image(mask_cond)
            cond = torch.cat([cond, mask_cond], dim=1)

        if self.use_scene_image_cond:
            assert "scene_image" in kwargs, "Scene image is not provided"
            scene_image = kwargs.pop("scene_image")
            scene_image = scene_image.expand(cond.shape[0], *scene_image.shape[1:])
            scene_cond = self.encode_image(scene_image)
            cond = torch.cat([cond, scene_cond], dim=1)

        if self.use_vggt_feature_cond:
            assert "vggt_feature" in kwargs, "VGGT feature is not provided"
            vggt_feature = kwargs.pop("vggt_feature")
            vggt_feature = vggt_feature.expand(cond.shape[0], *vggt_feature.shape[1:])
            vggt_feature = torch.cat([vggt_feature[..., :1024], vggt_feature[..., 1024:]], dim=1)
            cond = torch.cat([cond, vggt_feature], dim=1)

        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
