#!/usr/bin/env python
import os
import sys
import logging
import random
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Optional

from systems import SceneSystem
from configs import SceneEvaluationConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_name="scene_evaluation", config_path="configs/test")
def main(cfg: SceneEvaluationConfig):
    logger.info(f"Evaluation config:\n{OmegaConf.to_yaml(cfg)}")
    
    set_seed(cfg.seed)
    
    if not cfg.system.input_dir or not os.path.exists(cfg.system.input_dir):
        logger.error(f"Invalid input directory: {cfg.system.input_dir}")
        return
    
    if not cfg.system.output_dir or not os.path.exists(cfg.system.output_dir):
        logger.error(f"Invalid output directory: {cfg.system.output_dir}")
        return
    
    os.makedirs(cfg.system.save_dir, exist_ok=True)
    
    logger.info("Initializing evaluation system...")
    system = SceneSystem(cfg.system)
    system.setup()
    
    logger.info("Starting evaluation...")
    system.test_directory()
    
    system.on_test_end()
    logger.info(f"Evaluation finished. Results saved to: {cfg.system.save_dir}")


if __name__ == "__main__":
    main()
