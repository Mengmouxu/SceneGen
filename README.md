# <img src="./assets/icon.png" height="32" style="vertical-align:middle;"> SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass

This repository contains the official PyTorch implementation of SceneGen: https://arxiv.org/abs/2508.15769/. Feel free to reach out for discussions! 

**Now the Inference Code and Pretrained Models are released!**

<div align="center">
   <img src="./assets/SceneGen.png">
</div>

## 🌟 Some Information
[Project Page](https://mengmouxu.github.io/SceneGen/) $\cdot$ [Paper](https://arxiv.org/abs/2508.15769/) $\cdot$ [Checkpoints](https://huggingface.co/haoningwu/SceneGen/)

## ⏩ News
- [2025.8] The inference code and checkpoints are released.
- [2025.8] Our pre-print paper has been released on arXiv.

## 📦 Installation & Pretrained Models

### Prerequisites
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA A100 and RTX 3090 GPUs.
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 12.1.
  - Python version 3.8 or higher is required. 

### Installation Steps
1. Clone the repo:
    ```sh
    git clone https://github.com/Mengmouxu/SceneGen.git
    cd SceneGen
    ```

2. Install the dependencies:
    Create a new conda environment named `scenegen` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast --demo
    ```
    The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.

### Pretrained Models
1. First, create a directory in the SceneGen folder to store the checkpoints:
    ```sh
    mkdir -p checkpoints
    ```
2. Download the pretrained models for **SAM2-Hiera-Large** and **VGGT-1B** from [SAM2](https://huggingface.co/facebook/sam2-hiera-large/) and [VGGT](https://huggingface.co/facebook/VGGT-1B/), then place them in the `checkpoints` directory. (**SAM2** installation and its checkpoints are required for interactive generation with segmentation.)
3. Download our pretrained SceneGen model from [here](https://huggingface.co/haoningwu/SceneGen/) and place it in the `checkpoints` directory as follows:
    ```
    SceneGen/
    ├── checkpoints/
    │   ├── sam2-hiera-large
    │   ├── VGGT-1B
    │   └── scenegen
    |       ├──ckpts
    |       └──pipeline.json
    └── ...
    ```
## 💡 Inference
We provide two scripts for inference: `inference.py` for batch processing and `interactive_demo.py` for an interactive Gradio demo.

### Interactive Demo
This script launches a Gradio web interface for interactive scene generation.
- **Features**: It uses SAM2 for interactive image segmentation, allows for adjusting various generation parameters, and supports scene generation from single or multiple images.
- **Usage**:
  ```sh
  python interactive_demo.py
  ```
  > ## 🚀 Quick Start Guide
  >
  > ### 📷 Step 1: Input & Segment
  > 1.  **Upload your scene image.**
  > 2.  **Use the mouse to draw bounding boxes** around objects.
  > 3.  Click **"Run Segmentation"** to segment objects.
  > > *※ For multi-image generation: maintain consistent object annotation order across all images.*
  >
  > ### 🗃️ Step 2: Manage Cache
  > 1.  Click **"Add to Cache"** when satisfied with the segmentation.
  > 2.  Repeat Step 1-2 for multiple images.
  > 3.  Use **"Delete Selected"** or **"Clear All"** to manage cached images.
  >
  > ### 🎮 Step 3: Generate Scene
  > 1.  Adjust generation parameters (optional).
  > 2.  Click **"Generate 3D Scene"**.
  > 3.  Download the generated GLB file when ready.
  >
  > **💡 Pro Tip:**  Try the examples below to get started quickly!

### Pre-segmented Image Inference
This script processes a directory of pre-segmented images.
- **Input**: The input folder structure should be similar to `assets/masked_image_test`, containing segmented scene images.
- **Visualization**: For scenes with ground truth data, you can use the `--gradio` flag to launch a Gradio interface that visualizes both the ground truth and the generated model. We provide data from the 3D-FUTURE test set as a demonstration.
- **Usage**:
  ```sh
  python inference.py --gradio
  ```

## 📚 Dataset
To be updated soon...

## 🏋️‍♂️ Training
To be updated soon...

## Evaluation
To be updated soon...

## 📜 Citation
If you use this code and data for your research or project, please cite:

    @article{meng2025scenegen,
      author    = {Meng, Yanxu and Wu, Haoning and Zhang, Ya and Xie, Weidi},
      title     = {SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass},
      journal   = {arXiv preprint arXiv:2508.15769},
      year      = {2025},
    }

## TODO
- [x] Release Paper
- [x] Release Checkpoints & Inference Code
- [ ] Release Training Code
- [ ] Release Evaluation Code
- [ ] Release Data Processing Code

## Acknowledgements
Many thanks to the code bases from [TRELLIS](https://github.com/microsoft/TRELLIS), [DINOv2](https://github.com/facebookresearch/dinov2), and [VGGT](https://github.com/facebookresearch/vggt).

## Contact
If you have any questions, please feel free to contact [meng-mou-xu@sjtu.edu.cn](mailto:meng-mou-xu@sjtu.edu.cn) and [haoningwu3639@gmail.com](mailto:haoningwu3639@gmail.com).