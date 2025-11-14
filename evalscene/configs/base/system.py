from dataclasses import dataclass, field
from typing import List, Optional, Union
from omegaconf import MISSING


@dataclass
class BaseSystemConfig:
    input_dir: str = MISSING
    output_dir: str = MISSING
    save_dir: str = "outputs/eval_results"
    
    metrics: List[str] = field(default_factory=lambda: [
        "scene_cd", "scene_fscore", "object_cd", "object_fscore", "iou_bbox"
    ])
    device: str = "cuda"


@dataclass
class SceneSystemConfig(BaseSystemConfig):
    input_format: str = "glb"
    gt_format: str = "glb"
    
    num_points: int = 20000
    
    eval_scene_level: bool = True
    eval_object_level: bool = True
    use_icp: bool = True
    icp_max_iterations: int = 50
    icp_tolerance: float = 1e-5
    fscore_threshold: float = 0.1
