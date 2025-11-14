import os
import torch
import hydra
import logging
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class BaseSystem:
    @dataclass
    class Config:
        input_dir: str = ""
        output_dir: str = ""
        save_dir: str = ""
        metrics: List[str] = None
        device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._setup_directories()
        logger.info(f"System initialized, device: {cfg.device}")

    def _setup_directories(self):
        if self.cfg.save_dir:
            os.makedirs(self.cfg.save_dir, exist_ok=True)
            logger.info(f"Results will be saved to: {self.cfg.save_dir}")

    def setup(self):
        pass

    def compute_metrics(self, pred_data, gt_data, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the compute_metrics method")

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Subclasses must implement the test_step method")

    def on_test_end(self):
        logger.info("Evaluation finished")

    def get_save_dir(self) -> str:
        return self.cfg.save_dir

    def save_metrics_to_csv(self, metrics: Dict[str, Any], filename: str = "metrics.csv"):
        metrics_dict = {k: [v.item() if torch.is_tensor(v) else v] for k, v in metrics.items()}
        df = pd.DataFrame(metrics_dict)
        if not filename.endswith('.csv'):
            filename += '.csv'
        save_path = os.path.join(self.get_save_dir(), filename)
        df.to_csv(save_path, index=False)
        logger.info(f"Metrics saved to: {save_path}")

        logger.info("Evaluation Metrics:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v.item() if torch.is_tensor(v) else v}")
