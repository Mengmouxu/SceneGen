import os
import copy
from functools import partial
from contextlib import nullcontext

from ..modules.transformer.modulated import ModulatedTransformerCrossBlock

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .utils import *
from .base import Trainer
from ..utils.general_utils import *
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils
from safetensors.torch import safe_open, load_file
from peft import LoraConfig, get_peft_model


import re


class BasicTrainer(Trainer):
    """
    Trainer for basic training loop.
    
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
    """

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        if not self.vggt_mixin:
            lines.append(f'  - Batch size: {self.batch_size}')
            lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        else:
            lines.append(f'  - Max batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Batch split: {self.batch_split}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.lr_scheduler_config is not None:
            lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        if self.elastic_controller_config is not None:
            lines.append(f'  - Elastic memory: {indent(str(self.elastic_controller), 2)}')
        if self.grad_clip is not None:
            lines.append(f'  - Gradient clip: {indent(str(self.grad_clip), 2)}')
        lines.append(f'  - EMA rate: {self.ema_rate}')
        lines.append(f'  - FP16 mode: {self.fp16_mode}')
        return '\n'.join(lines)
            
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=True
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models

        # Build master params
        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()]
        , [])
        if self.fp16_mode == 'amp':
            self.master_params = self.model_params
            self.scaler = torch.GradScaler() if self.fp16_mode == 'amp' else None
        elif self.fp16_mode == 'inflat_all':
            self.master_params = make_master_params(self.model_params)
            self.fp16_scale_growth = self.fp16_scale_growth
            self.log_scale = 20.0
        elif self.fp16_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'FP16 mode {self.fp16_mode} is not implemented.')

        # Build EMA params
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        if hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **self.optimizer_config['args'])
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            assert any([isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        """
        if self.fp16_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    # def _state_dicts_to_master_params(self, master_params, state_dicts):
    #     """
    #     Convert a state_dict to master params.
    #     """
    #     master_params_names = sum(
    #         [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
    #     , [])
    #     params = [state_dicts[name][param_name] for name, param_name in master_params_names]
    #     if self.fp16_mode == 'inflat_all':
    #         model_params_to_master_params(params, master_params)
    #     else:
    #         for i, param in enumerate(params):
    #             master_params[i].data.copy_(param.data)

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params with non-strict loading support.
        """
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        
        params = []
        missing_keys = []
        for name, param_name in master_params_names:
            if name in state_dicts and param_name in state_dicts[name]:
                params.append(state_dicts[name][param_name])
            else:
                missing_keys.append(f"{name}.{param_name}")
                for model_name, model in self.models.items():
                    if model_name == name:
                        for n, p in model.named_parameters():
                            if n == param_name and p.requires_grad:
                                params.append(p.data.clone())
                                break
        
        if missing_keys and self.is_master:
            print(f"Warning: Missing {len(missing_keys)} keys in state_dicts (non-strict loading)")
            if len(missing_keys) < 10:
                print(f"Missing keys: {missing_keys}")
            else:
                print(f"First 10 missing keys: {missing_keys[:10]}...")
        
        if self.fp16_mode == 'inflat_all':
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                if i < len(master_params):
                    master_params[i].data.copy_(param.data)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')
            
        model_ckpts = {}
        for name, model in self.models.items():
            model_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')), map_location=self.device, weights_only=True)
            model_ckpts[name] = model_ckpt
            model.load_state_dict(model_ckpt)

            if self.freeze_weights or self.unfreeze_weights:
                frozen_count = 0
                total_count = 0
                
                for param in model.parameters():
                    param.requires_grad = True
                    
                if self.unfreeze_weights is not None:
                    if self.is_master:
                        print(f' - Unfreezing weights of {self.unfreeze_weights}')
                    for name, param in model.named_parameters():
                        if any(re.search(nfreeze, name) for nfreeze in self.unfreeze_weights):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                            frozen_count += param.numel()
                        total_count += param.numel()
                elif self.freeze_weights:
                    if self.is_master:
                        print(f' - Freezing weights of {self.freeze_weights}')
                    for name, param in model.named_parameters():
                        if any(re.search(freeze, name) for freeze in self.freeze_weights):
                            param.requires_grad = False
                            frozen_count += param.numel()
                        else:
                            param.requires_grad = True
                        total_count += param.numel()
                        
                if self.is_master:
                    print(f' - {total_count} parameters in {model.__class__.__name__}.')
                    print(f' - {total_count - frozen_count} parameters are trainable ({100 * (total_count - frozen_count) / total_count:.2f}%).')
                    print(f' - {frozen_count} parameters are frozen ({100 * frozen_count / total_count:.2f}%).')
                    
            if self.fp16_mode == 'inflat_all':
                model.convert_to_fp16()

        if self.freeze_weights or self.unfreeze_weights:
            self.model_params = sum(
                [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()]
            , [])
        
            if self.fp16_mode == 'amp':
                self.master_params = self.model_params
            elif self.fp16_mode == 'inflat_all':
                self.master_params = make_master_params(self.model_params)
            elif self.fp16_mode is None:
                self.master_params = self.model_params
                
            if hasattr(torch.optim, self.optimizer_config['name']):
                self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(
                    self.master_params, **self.optimizer_config['args'])
            else:
                self.optimizer = globals()[self.optimizer_config['name']](
                    self.master_params, **self.optimizer_config['args'])
                
            if self.is_master:
                self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]
        
        self._state_dicts_to_master_params(self.master_params, model_ckpts)

        del model_ckpts

        if self.is_master and not self.freeze_weights and not self.unfreeze_weights:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt = torch.load(os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt'), map_location=self.device, weights_only=True)
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts
        
        misc_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')), map_location=torch.device('cpu'), weights_only=False)
        if not self.freeze_weights and not self.unfreeze_weights:
            self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        self.step = misc_ckpt['step']
        # self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.fp16_mode == 'amp':
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.fp16_mode == 'inflat_all':
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, 'save() should be called only by the rank 0 process.'
        print(f'\nSaving checkpoint at step {self.step}...', end='')
        
        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            torch.save(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))
        
        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                torch.save(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt'))
                
        if not self.vggt_mixin:
            misc_ckpt = {
                'optimizer': self.optimizer.state_dict(),
                'step': self.step,
                'data_sampler': self.data_sampler.state_dict(),
            }
        else:
            misc_ckpt = {
                'optimizer': self.optimizer.state_dict(),
                'step': self.step,
            }

        if self.fp16_mode == 'amp':
            misc_ckpt['scaler'] = self.scaler.state_dict()
        elif self.fp16_mode == 'inflat_all':
            misc_ckpt['log_scale'] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
        torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        print(' Done.')

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                # model_ckpt = torch.load(read_file_dist(finetune_ckpt[name]), map_location=self.device, weights_only=True)
                # model_ckpt = torch.load(read_file_dist(finetune_ckpt[name]), map_location=self.device, weights_only=False)

                checkpoint_path = finetune_ckpt[name]
            
                if checkpoint_path.endswith('.safetensors'):
                    if self.is_master:
                        print(f" - Loading {name} from safetensors file: {checkpoint_path}")
                    model_ckpt = {}
                    with safe_open(checkpoint_path, framework="pt") as f:
                        for k in f.keys():
                            model_ckpt[k] = f.get_tensor(k).to(self.device)
                else:
                    checkpoint_path = read_file_dist(checkpoint_path)
                    if self.is_master:
                        print(f" - Loading {name} using torch.load: {checkpoint_path}")
                    try:
                        model_ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                    except Exception as e:
                        if self.is_master:
                            print(f" - Warning: Failed to load with weights_only=True: {e}")
                            print("Attempting unsafe load (only do this if checkpoint is from trusted source)")
                        model_ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                if self.peft_ckpt_path is not None and os.path.exists(self.peft_ckpt_path):
                    # First, load the base model weights
                    model.load_state_dict(model_ckpt, strict=False)

                    # Define and apply PEFT config
                    peft_config = LoraConfig(
                                    r=64,
                                    lora_alpha=128,
                                    lora_dropout=0.0,
                                    target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
                                )
                    if self.is_master:
                        print(f' - Applying PEFT config: {peft_config}')
                    model = get_peft_model(model, peft_config)
                    
                    # Load the PEFT adapter weights
                    if self.is_master:
                        print(f' - Loading PEFT adapter weights from: {self.peft_ckpt_path}')
                    adapter_weights = torch.load(self.peft_ckpt_path, map_location=self.device)
                    model.load_state_dict(adapter_weights, strict=False)

                    # Update model_ckpt to include all weights for later processing
                    model_ckpt = model.state_dict()
                else:
                    # Original behavior if no PEFT checkpoint is provided
                    for k, v in model_ckpt.items():
                        if k in model_state_dict and model_ckpt[k].shape != model_state_dict[k].shape:
                            if self.is_master:
                                print(f' - Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                            model_ckpt[k] = model_state_dict[k]
                    model.load_state_dict(model_ckpt, strict=False)
                


                if self.init_global_params:
                    if self.is_master:
                        print(f' - Initializing global modules from local for model "{name}"...')
                    
                    copied_count = 0
                    copied_parameters = 0

                    for module in model.modules():
                        if isinstance(module, ModulatedTransformerCrossBlock) and module.use_global:
                            with torch.no_grad():
                                # 1. 复制 adaLN_modulation -> adaLN_modulation_global
                                source_mod_weight = module.adaLN_modulation[1].weight
                                target_mod_weight = module.adaLN_modulation_global[1].weight
                                target_mod_weight.copy_(source_mod_weight[:target_mod_weight.size(0), :])
                                if module.adaLN_modulation[1].bias is not None:
                                    source_mod_bias = module.adaLN_modulation[1].bias
                                    target_mod_bias = module.adaLN_modulation_global[1].bias
                                    target_mod_bias.copy_(source_mod_bias[:target_mod_bias.size(0)])

                                # 2. 复制 self_attn -> global_attn
                                module.global_attn.to_qkv.weight.copy_(module.self_attn.to_qkv.weight)
                                if module.self_attn.to_qkv.bias is not None:
                                    module.global_attn.to_qkv.bias.copy_(module.self_attn.to_qkv.bias)
                                
                                module.global_attn.to_out.weight.copy_(module.self_attn.to_out.weight)
                                if module.self_attn.to_out.bias is not None:
                                    module.global_attn.to_out.bias.copy_(module.self_attn.to_out.bias)

                                # 3. 复制 cross_attn -> global_cross_attn
                                module.global_cross_attn.to_q.weight.copy_(module.cross_attn.to_q.weight)
                                if module.cross_attn.to_q.bias is not None:
                                    module.global_cross_attn.to_q.bias.copy_(module.cross_attn.to_q.bias)

                                module.global_cross_attn.to_kv.weight.copy_(module.cross_attn.to_kv.weight)
                                if module.cross_attn.to_kv.bias is not None:
                                    module.global_cross_attn.to_kv.bias.copy_(module.cross_attn.to_kv.bias)

                                module.global_cross_attn.to_out.weight.copy_(module.cross_attn.to_out.weight)
                                if module.cross_attn.to_out.bias is not None:
                                    module.global_cross_attn.to_out.bias.copy_(module.cross_attn.to_out.bias)
                                
                                # 4. 统计复制的参数量
                                copied_parameters += module.adaLN_modulation_global[1].weight.numel()
                                if module.adaLN_modulation_global[1].bias is not None:
                                    copied_parameters += module.adaLN_modulation_global[1].bias.numel()
                                
                                copied_parameters += sum(p.numel() for p in module.global_attn.parameters())
                                copied_parameters += sum(p.numel() for p in module.global_cross_attn.parameters())
                            
                            copied_count += 1
                    
                    if self.is_master:
                        print(f'   - Done. Copied weights for {copied_count} ModulatedTransformerCrossBlock modules.')
                        print(f'   - Copied {copied_parameters} parameters to global modules.')

                model_ckpts[name] = model_ckpt
                # For finetuning with different model architectures
                model.load_state_dict(model_ckpt, strict=False)
                # Make all model parameters trainable
                for param in model.parameters():
                    param.requires_grad = True
                if self.is_master:
                    print(f' - All parameters in {model.__class__.__name__} are set to trainable.')
                
                if self.freeze_weights or self.unfreeze_weights:
                    frozen_count = 0
                    total_count = 0
                    if self.unfreeze_weights is not None:
                        if self.is_master:
                            print(f' - Unfreezing weights of {self.unfreeze_weights}')
                        for name, param in model.named_parameters():
                            if any(re.search(nfreeze, name) for nfreeze in self.unfreeze_weights):
                                param.requires_grad = True
                                # print(f'Unfreezing {name}')
                            else:
                                param.requires_grad = False
                                frozen_count += param.numel()
                            total_count += param.numel()
                    else:
                        if self.freeze_weights:
                            for name, param in model.named_parameters():
                                if any(re.search(freeze, name) for freeze in self.freeze_weights):
                                    param.requires_grad = False
                                    frozen_count += param.numel()
                                else:
                                    param.requires_grad = True
                                total_count += param.numel()

                    if self.is_master:
                        print(f' - {total_count} parameters in {model.__class__.__name__}.')
                        print(f' - {total_count - frozen_count} parameters are trainable.')
                        if self.unfreeze_weights:
                            print(f'    - with {self.unfreeze_weights} unfrozen and proportion of {100 * (total_count - frozen_count) / total_count:.2f}%.')
                        print(f' - {frozen_count} parameters are frozen.')
                        if self.freeze_weights:
                            print(f'    - with {self.freeze_weights} frozen and proportion of {100 * frozen_count / total_count:.2f}%.')
                    
                self.model_params = sum(
                    [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()]
                , [])
                
                if self.fp16_mode == 'amp':
                    self.master_params = self.model_params
                elif self.fp16_mode == 'inflat_all':
                    self.master_params = make_master_params(self.model_params)
                elif self.fp16_mode is None:
                    self.master_params = self.model_params
                    
                if hasattr(torch.optim, self.optimizer_config['name']):
                    self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(
                        self.master_params, **self.optimizer_config['args'])
                else:
                    self.optimizer = globals()[self.optimizer_config['name']](
                        self.master_params, **self.optimizer_config['args'])
                    
                if self.is_master:
                    self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

                if self.fp16_mode == 'inflat_all':
                    model.convert_to_fp16()
                
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)

        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i:i+sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'
        except AssertionError as e:
            if self.is_master:
                print(f'\n\033[91mError: {e}\033[0m')
                print('DDP check failed.')
            raise e

        dist.barrier()
        if self.is_master:
            print('Done.')

    def run_step(self, data_list):
        """
        Run a training step.
        """
        step_log = {'loss': {}, 'status': {}}
        amp_context = partial(torch.autocast, device_type='cuda') if self.fp16_mode == 'amp' else nullcontext
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        # Train
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = [self.training_models[name].no_sync for name in self.training_models] if i != len(data_list) - 1 and self.world_size > 1 else [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss['loss'] / len(data_list)
                ## backward
                if self.fp16_mode == 'amp':
                    self.scaler.scale(l).backward()
                elif self.fp16_mode == 'inflat_all':
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            ## log
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
        ## gradient clip
        if self.grad_clip is not None:
            if self.fp16_mode == 'amp':
                self.scaler.unscale_(self.optimizer)
            elif self.fp16_mode == 'inflat_all':
                model_grads_to_master_grads(self.model_params, self.master_params)
                self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]['grad_norm'] = grad_norm.item()
        ## step
        if self.fp16_mode == 'amp':
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.fp16_mode == 'inflat_all':
            prev_scale = 2 ** self.log_scale
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                    self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
                self.optimizer.step()
                master_params_to_model_params(self.model_params, self.master_params)
                self.log_scale += self.fp16_scale_growth
            else:
                self.log_scale -= 1
        else:
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m') 
        if self.log_scale < 0:
            self.log_scale = 10.0
        ## adjust learning rate
        if self.lr_scheduler_config is not None:
            statuses[-1]['lr'] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        # Logs
        step_log['loss'] = dict_reduce(losses, lambda x: np.mean(x))
        step_log['status'] = dict_reduce(statuses, lambda x: np.mean(x), special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)})
        if self.elastic_controller_config is not None:
            step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()
            
        # Check grad and norm of each param
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    param_norms[name] = param.norm().item()
                    if param.grad is not None and torch.isfinite(param.grad).all():
                        param_grads[name] = param.grad.norm().item() / prev_scale
            step_log['param_norms'] = param_norms
            step_log['param_grads'] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

        return step_log
