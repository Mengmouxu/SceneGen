import os
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Dict, List, Optional, Iterator
from torch.utils.data import DataLoader
from .sparse_structure_latent import ImageConditionedSparseStructureLatentVGGT

class DynamicBatchSampler(Sampler):
    """
    Dynamic batch size sampler - Ensures consistent batch size across all GPUs in the same batch.
    
    Args:
        dataset: Instance of ImageConditionedSparseStructureLatentVGGT dataset
        batch_size: Maximum allowed batch size
        drop_last: Whether to drop incomplete batches
        shuffle: Whether to shuffle the data
        seed: Random seed
        distributed: Whether running in a distributed environment
    """
    def __init__(
        self,
        dataset,
        batch_size: int = 8, 
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        distributed: bool = False,
        assign_bs: Optional[List[int]] = None,
    ):
        self.dataset = dataset
        self.max_batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.distributed = distributed
        
        if self.distributed:
            self.rank = dist.get_rank()
            if self.rank == 0:
                print("Distributed training mode enabled for DynamicBatchSampler.")

            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            
        # Group dataset instances by batch size (bs)
        self.bs_groups = {}
        for idx, (_, _, _, bs) in enumerate(self.dataset.instances):
            if bs not in self.bs_groups:
                self.bs_groups[bs] = []
            self.bs_groups[bs].append(idx)
            
        # Record valid batch sizes
        self.valid_bs_values = sorted(list(self.bs_groups.keys()))
        
        # Remove groups exceeding the maximum batch size
        if self.max_batch_size > 0:
            self.valid_bs_values = [bs for bs in self.valid_bs_values if bs <= self.max_batch_size]
        
        if assign_bs is not None:
            # Ensure the assigned batch sizes are valid
            self.valid_bs_values = [bs for bs in self.valid_bs_values if bs in assign_bs]
        
        # Generate indices for the current epoch
        self._generate_batch_indices()
    
    def _generate_batch_indices(self):
        """Generate synchronized batch indices"""
        # Ensure all processes use the same random seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Synchronized batch structure: [(bs, [indices_rank0], [indices_rank1], ...), ...]
        self.synced_batches = []
        
        # Process each valid batch size
        for bs in self.valid_bs_values:
            indices = self.bs_groups[bs]
            if len(indices) == 0:
                continue
                
            if self.shuffle:
                # Ensure all processes use the same shuffle order
                indices_tensor = torch.tensor(indices)
                shuffle_idx = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices_tensor[i].item() for i in shuffle_idx]
            
            # Calculate the number of full batches
            num_samples = len(indices)
            samples_per_gpu = (num_samples // self.world_size) if self.distributed else num_samples
            num_full_batches = samples_per_gpu if not self.drop_last else (samples_per_gpu // self.world_size * self.world_size)
            
            # Only take samples that can form full batches
            usable_indices = indices[:num_full_batches * self.world_size] if self.distributed else indices[:num_full_batches]
            
            # Assign to different GPUs
            if self.distributed:
                batches_per_gpu = [[] for _ in range(self.world_size)]
                for i, idx in enumerate(usable_indices):
                    rank_id = i % self.world_size
                    batches_per_gpu[rank_id].append(idx)
                
                # Create synchronized batches
                # Ensure all GPUs in a batch have the same batch size
                for i in range(0, len(batches_per_gpu[0]), 1):
                    batch = []
                    for r in range(self.world_size):
                        if i < len(batches_per_gpu[r]):
                            batch.append([batches_per_gpu[r][i]])
                        else:
                            # If a rank doesn't have enough data, duplicate data from the first rank
                            batch.append([batches_per_gpu[0][0]])
                    self.synced_batches.append((bs, batch))
            else:
                # Non-distributed case
                batches = []
                for i in range(0, len(usable_indices), 1):
                    batches.append([usable_indices[i]])
                    
                for batch in batches:
                    self.synced_batches.append((bs, [batch]))
        
        # Shuffle the order of batches with different batch sizes
        if self.shuffle:
            random_order = torch.randperm(len(self.synced_batches), generator=g).tolist()
            self.synced_batches = [self.synced_batches[i] for i in random_order]
    
    def __iter__(self) -> Iterator:
        """Return an iterator"""
        # In distributed mode, each rank only gets its corresponding indices
        if self.distributed:
            for bs, batches in self.synced_batches:
                yield batches[self.rank]
        else:
            for bs, batches in self.synced_batches:
                yield batches[0]
    
    def __len__(self) -> int:
        """Return the number of batches"""
        return len(self.synced_batches)
    
    def set_epoch(self, epoch: int) -> None:
        """Set a new epoch and regenerate batches"""
        self.epoch = epoch
        self._generate_batch_indices()

def create_dynamic_batch_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    shuffle=True,
    drop_last=True,
    distributed=False,
    seed=0,
    **kwargs
):
    """
    Create a dataloader that supports dynamic batch sizes.
    
    Args:
        dataset: Instance of ImageConditionedSparseStructureLatentVGGT dataset
        batch_size: Maximum batch size per GPU
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        distributed: Whether running in a distributed environment
        seed: Random seed
        **kwargs: Additional arguments passed to DataLoader
    """
    # Initialize distributed environment
    if distributed and not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized")
        
    # Create custom sampler
    sampler = DynamicBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        distributed=distributed
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        **kwargs
    )

    return loader

class CyclicLoader:
    def __init__(self, dataloader, start_epoch: int = 0):
        self.dataloader = dataloader
        self.epoch = start_epoch
        self._iterator= None
        self._init_iterator()
    
    def _init_iterator(self):
        if hasattr(self.dataloader, 'batch_sampler') and hasattr(self.dataloader.batch_sampler, 'set_epoch'):
            self.dataloader.batch_sampler.set_epoch(self.epoch)
        elif hasattr(self.dataloader, 'sampler') and hasattr(self.dataloader.sampler, 'set_epoch'):
            self.dataloader.sampler.set_epoch(self.epoch)
        self._iterator = iter(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self._init_iterator()
            return next(self._iterator)

def init_distributed(force=False):
    """
    Initialize the distributed environment.
    
    Args:
        force: If True, attempts to use all available GPUs for distributed training.
        
    Returns:
        bool: Whether the distributed environment was successfully initialized.
        
    Usage:
    1. Multi-GPU on a single machine: Use `torchrun --nproc_per_node={number_of_gpus} your_script.py` to start.
    2. Manual force: Set force=True to set basic environment variables and attempt to use all GPUs.
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPU count: {gpu_count}")
        
        # Check if distributed environment variables are set
        env_init = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        
        # If force is enabled or environment variables are already set
        if force or env_init:
            if force and not env_init:
                # Set default distributed environment variables
                os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
                os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
                
                if 'LOCAL_RANK' in os.environ:
                    # Environment provided by torchrun or torch.distributed.launch
                    local_rank = int(os.environ['LOCAL_RANK'])
                    os.environ['RANK'] = str(local_rank)
                    os.environ['WORLD_SIZE'] = str(gpu_count)
                else:
                    # Single-process force mode - Warning: This will only use one GPU
                    print("Warning: No distributed launcher environment detected. Only one GPU will be used.")
                    print("Please use 'torchrun --nproc_per_node={} your_script.py' to use all GPUs".format(gpu_count))
                    os.environ['RANK'] = '0'
                    os.environ['WORLD_SIZE'] = str(gpu_count)
                    os.environ['LOCAL_RANK'] = '0'
            
            # Initialize process group
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            print(f"Process {os.environ['RANK']}/{os.environ['WORLD_SIZE']} using GPU {local_rank}")
            return True
    
    print("Failed to initialize distributed environment. Single GPU mode will be used.")
    return False