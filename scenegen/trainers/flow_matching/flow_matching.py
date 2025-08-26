from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import os
import json
from scipy.spatial.transform import Rotation as R

from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin, ImageConditionedVGGTMixin
from ...datasets.dataset_utils import DynamicBatchSampler, CyclicLoader
from ... import models
from ...utils.dist_utils import read_file_dist

class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        smooth_scale: float = 0.1,
        collision_smooth_scale: float = 0.1,
        trans_weight: float = 2,
        rot_weight: float = 3,
        scale_weight: float = 2,
        position_weight_min_ratio: float = 0.2,
        position_weight_max_ratio: float = 1.0,
        use_collision_loss: bool = False,
        collision_loss_weight: float = 1.0,
        collision_resolution: int = 128,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min
        self.smooth_scale = smooth_scale
        self.collision_smooth_scale = collision_smooth_scale
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.scale_weight = scale_weight
        self.position_weight_min_ratio = position_weight_min_ratio
        self.position_weight_max_ratio = position_weight_max_ratio

        self.use_collision_loss = use_collision_loss
        self.collision_loss_weight = collision_loss_weight
        self.collision_resolution = collision_resolution
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        self._loading_ss_dec()
    
    def _loading_ss_dec(self):
        if self.use_collision_loss:
            if self.ss_dec_path is not None:
                cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
                decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
                ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
                decoder.load_state_dict(torch.load(read_file_dist(ckpt_path), map_location='cpu', weights_only=True))
            else:
                decoder = models.from_pretrained(self.pretrained_ss_dec)
            self.ss_dec = decoder.cuda().eval()
            return
        else:
            self.ss_dec = None
            return
    
    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs):
        """
        Get the sampler for the diffusion process.
        """
        if not self.vggt_mixin:
            return samplers.FlowEulerSampler(self.sigma_min)
        else:
            return samplers.FlowEulerSamplerVGGT(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        if self.vggt_mixin:
            positions = kwargs.pop('positions')
        
        if self.vggt_mixin:
            pred, pred_positions = self.training_models['denoiser'](x_t, t * 1000, cond)
            # Add position prediction loss   
            assert pred.shape == noise.shape == x_0.shape
            target = self.get_v(x_0, noise, t)
            
            terms = edict()
            terms["mse"] = F.mse_loss(pred, target)

            if x_0.shape[0] > 1:
                if self.use_collision_loss:
                    if self.ss_dec is None:
                        raise ValueError("SS Decoder is not loaded. Please set use_collision_loss to False or load the SS Decoder.")
                    # Use SS Decoder to compute collision loss
                    # x_t = (1 - t) * x_0 + σ(t) * noise
                    # v = (1 - σ_min) * noise - x_0
                    # noise = (v + x_0) / (1 - σ_min)
                    # x_t = (1 - t) * x_0 + σ(t) * (v + x_0) / (1 - σ_min)
                    # x_t = (1 - t) * x_0 + σ(t) * v / (1 - σ_min) + σ(t) * x_0 / (1 - σ_min)
                    # x_t = x_0 * [(1 - t) + σ(t) / (1 - σ_min)] + σ(t) * v / (1 - σ_min)
                    # x_0 = [x_t - σ(t) * v / (1 - σ_min)] / [(1 - t) + σ(t) / (1 - σ_min)]

                    _t_reshaped = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
                    _sigma_t = self.sigma_min + (1 - self.sigma_min) * _t_reshaped
                    
                    numerator = x_t - _sigma_t * pred / (1 - self.sigma_min)
                    denominator = (1 - _t_reshaped) + _sigma_t / (1 - self.sigma_min)
                    x_0_pred = numerator / denominator
                    
                    scene_positions = []
                    assets_points_num = []
                    for i in range(x_0_pred.shape[0]):
                        coords_i = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
                        resolution = x_0.shape[-1]
                        org_position = coords_i.float() / resolution
                        if i != 0:
                            translation = positions[i, 0:3].float()
                            rotation = positions[i, 3:7].float()
                            scale = positions[i, 7].float()
                        else:
                            translation = torch.tensor([0.0, 0.0, 0.0], device=x_0.device, dtype=x_0.dtype)
                            rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=x_0.device, dtype=x_0.dtype)
                            scale = torch.tensor(1.0, device=x_0.device, dtype=x_0.dtype)

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
                        assets_points_num.append(final_position.shape[0])

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

                    # Define a resolution for voxelization
                    voxel_resolution = self.collision_resolution

                    # Initialize a grid that counts how many objects occupy each voxel
                    # Create voxel grid to track occupancy counts
                    voxel_counts = torch.zeros((voxel_resolution, voxel_resolution, voxel_resolution), 
                                            dtype=torch.int, device=scene_positions.device)

                    # Convert scene positions to voxel coordinates
                    voxel_indices = (scene_positions * (voxel_resolution - 1)).long()
                    voxel_indices = torch.clamp(voxel_indices, 0, voxel_resolution - 1)

                    # Track starting index for each object
                    start_idx = 0
                    for obj_idx, num_points in enumerate(assets_points_num):
                        if obj_idx == 0:  # Skip the reference object
                            start_idx += num_points
                            continue
                        
                        # Get and process points for this object
                        end_idx = start_idx + num_points
                        obj_indices = voxel_indices[start_idx:end_idx]
                        
                        # Directly update voxel occupancy (using unique to count each voxel only once per object)
                        unique_indices = torch.unique(obj_indices, dim=0)
                        voxel_counts[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] += 1
                        
                        start_idx = end_idx

                    # Calculate collision metrics
                    collision_voxels = (voxel_counts > 1).sum()
                    total_occupied_voxels = (voxel_counts > 0).sum()
                    collision_ratio = collision_voxels.float() / (total_occupied_voxels.float() + 1e-8)
                    terms['collision_ratio'] = collision_ratio


                scene_scale = torch.max(positions[0:, :3] + positions[0:, 7].unsqueeze(-1) / 2, dim=0).values - torch.min(positions[0:, :3] - positions[0:, 7].unsqueeze(-1) / 2, dim=0).values
                min_val = torch.tensor(1e-2, device=scene_scale.device, dtype=scene_scale.dtype)
                scene_scale = torch.max(scene_scale, min_val)
            
                trans_loss = F.smooth_l1_loss(pred_positions[1:, :3] / scene_scale / self.smooth_scale, positions[1:, :3] / scene_scale / self.smooth_scale) * self.smooth_scale
                rot_loss = F.smooth_l1_loss(pred_positions[1:, 3:7] / self.smooth_scale, positions[1:, 3:7] / self.smooth_scale) * self.smooth_scale
                scale_loss = F.smooth_l1_loss(pred_positions[1:, 7:] / self.smooth_scale, positions[1:, 7:] / self.smooth_scale) * self.smooth_scale
                terms['trans_loss'] = F.l1_loss(pred_positions[1:, :3], positions[1:, :3]).detach()
                terms['rot_loss'] = F.l1_loss(pred_positions[1:, 3:7], positions[1:, 3:7]).detach()
                terms['scale_loss'] = F.l1_loss(pred_positions[1:, 7:], positions[1:, 7:]).detach()
                terms["pos_loss"] = trans_loss * self.trans_weight + rot_loss * self.rot_weight + scale_loss * self.scale_weight
                if self.use_collision_loss:
                    terms["pos_loss"] += F.smooth_l1_loss(
                        terms['collision_ratio'] / self.collision_smooth_scale,
                        torch.tensor(0.0, device=terms['collision_ratio'].device, dtype=terms['collision_ratio'].dtype).view_as(terms['collision_ratio'])
                    ) * self.collision_smooth_scale * self.collision_loss_weight
                
                mse_norm = terms["mse"].detach()
                pos_norm = terms["pos_loss"].detach()
                ratio = mse_norm / (pos_norm + 1e-8)
                
                if not hasattr(self, 'ema_pos_weight'):
                    self.ema_pos_weight = ratio.detach()
                else:
                    decay = 0.99
                    self.ema_pos_weight = decay * self.ema_pos_weight + (1 - decay) * ratio.detach()

                pos_weight = torch.clamp(self.ema_pos_weight, 
                                        min=self.position_weight_min_ratio, 
                                        max=self.position_weight_max_ratio)
                # pos_weight = torch.clamp(ratio, min=self.position_weight_min_ratio, max=self.position_weight_max_ratio)
                terms["pos_weight"] = pos_weight

                # Combine velocity and position losses
                terms["loss"] = terms["mse"] + terms["pos_loss"] * pos_weight
            else:
                terms["loss"] = terms["mse"]
            
        else:
            pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
            assert pred.shape == noise.shape == x_0.shape
            target = self.get_v(x_0, noise, t)
            terms = edict()
            terms["mse"] = F.mse_loss(pred, target)
            terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        if not self.vggt_mixin:
            dataloader = DataLoader(
                copy.deepcopy(self.dataset),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            )
            # inference
            sampler = self.get_sampler()
            sample_gt = []
            sample = []
            cond_vis = []
            for i in range(0, num_samples, batch_size):
                batch = min(batch_size, num_samples - i)
                data = next(iter(dataloader))
                data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
                noise = torch.randn_like(data['x_0'])
                sample_gt.append(data['x_0'])
                cond_vis.append(self.vis_cond(**data))
                del data['x_0']
                args = self.get_inference_cond(**data)
                res = sampler.sample(
                    self.models['denoiser'],
                    noise=noise,
                    **args,
                    steps=50, cfg_strength=3.0, verbose=verbose,
                )
                sample.append(res.samples)

            sample_gt = torch.cat(sample_gt, dim=0)
            sample = torch.cat(sample, dim=0)
            sample_dict = {
                'sample_gt': {'value': sample_gt, 'type': 'sample'},
                'sample': {'value': sample, 'type': 'sample'},
            }
            sample_dict.update(dict_reduce(cond_vis, None, {
                'value': lambda x: torch.cat(x, dim=0),
                'type': lambda x: x[0],
            }))
            
            return sample_dict
        else:
            data_sampler = DynamicBatchSampler(
                self.dataset,
                batch_size=num_samples,
                shuffle=True,
                seed=np.random.randint(0, 114514),
                assign_bs=[batch_size],
            )
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                num_workers=0,
                batch_sampler=data_sampler,
                collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            )

            # inference
            sampler = self.get_sampler()
            sample_gt = []
            pos_gt = []
            sample = []
            pos = []
            cond_vis = []
            for i in range(0, num_samples, batch_size):
                batch = min(batch_size, num_samples - i)
                data = next(iter(dataloader))
                for k in self.squeeze_data:
                    if k in data:
                        data[k] = data[k].squeeze(0)
                data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
                noise = torch.randn_like(data['x_0'])
                sample_gt.append(data['x_0'])
                pos_gt.append(data['positions'])
                cond_vis.append(self.vis_cond(**data))
                del data['x_0']
                del data['positions']
                args = self.get_inference_cond(**data)
                res = sampler.sample(
                    self.models['denoiser'],
                    noise=noise,
                    **args,
                    steps=50, cfg_strength=3.0, verbose=verbose,
                )
                sample.append(res.samples)
                pos.append(res.positions)

            sample_gt = torch.cat(sample_gt, dim=0)
            sample = torch.cat(sample, dim=0)
            pos_gt = torch.cat(pos_gt, dim=0)
            pos = torch.cat(pos, dim=0)
            sample_dict = {
                'sample_gt': {'value': sample_gt, 'type': 'sample'},
                'sample': {'value': sample, 'type': 'sample'},
                'pos_gt': {'value': pos_gt, 'type': 'position'},
                'pos': {'value': pos, 'type': 'position'},
            }
            sample_dict.update(dict_reduce(cond_vis, None, {
                'value': lambda x: torch.cat(x, dim=0),
                'type': lambda x: x[0],
            }))
            
            return sample_dict

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    def get_sampler(self, **kwargs):
        if not self.vggt_mixin:
            return samplers.FlowEulerCfgSampler(self.sigma_min)
        else:
            return samplers.FlowEulerCfgSamplerVGGT(self.sigma_min)
    pass


class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass

class ImageConditionedVGGTFlowMatchingCFGTrainer(ImageConditionedVGGTMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
        freeze_weights (list): List of weights to freeze.
        unfreeze_weights (list): List of weights to unfreeze.
    """
    pass