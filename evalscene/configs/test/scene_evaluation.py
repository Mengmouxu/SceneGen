from dataclasses import dataclass, field
from typing import List, Optional

from ..base import SceneSystemConfig


@dataclass
class SceneEvaluationConfig:
    system: SceneSystemConfig = field(default_factory=SceneSystemConfig)
    
    experiment_name: str = "scene_evaluation"
    seed: int = 42
    
    def __post_init__(self):
        if self.system.save_dir == "outputs/eval_results":
            self.system.save_dir = f"outputs/eval_results/{self.experiment_name}" 