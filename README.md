# <img src="./assets/icon.png" height="32" style="vertical-align:middle;"> SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass (3DV 2026)

This repository contains the official PyTorch implementation of SceneGen: https://arxiv.org/abs/2508.15769/. 

**Now the Training, Inference Code and Pretrained Models have all been released! Feel free to reach out for discussions!**

<div align="center">
   <img src="./assets/SceneGen.png">
</div>

## 🌟 Some Information
[Project Page](https://mengmouxu.github.io/SceneGen/) $\cdot$ [Paper](https://arxiv.org/abs/2508.15769/) $\cdot$ [Checkpoints](https://huggingface.co/haoningwu/SceneGen/)

## ⏩ News
- [2025.11] Glad to share that SceneGen has been accepted to 3DV 2026.
- [2025.9] Our training code with configs and data processing code are released.
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


https://github.com/user-attachments/assets/d0d53506-70cd-4bd3-a6ab-2f9b5b16f4d8


*Click the image above to watch the demo video*

### Pre-segmented Image Inference
This script processes a directory of pre-segmented images.
- **Input**: The input folder structure should be similar to `assets/masked_image_test`, containing segmented scene images.
- **Visualization**: For scenes with ground truth data, you can use the `--gradio` flag to launch a Gradio interface that visualizes both the ground truth and the generated model. We provide data from the 3D-FUTURE test set as a demonstration.
- **Usage**:
  ```sh
  python inference.py --gradio
  ```

## 📚 Dataset
To train and evaluate SceneGen, we use the [3D-FUTURE](https://tianchi.aliyun.com/dataset/98063) dataset. Please download and preprocess the dataset as follows:
1. Download the 3D-FUTURE dataset from [here](https://tianchi.aliyun.com/dataset/98063) which requires applying for access.
2. Follow the [TRELLIS](https://github.com/microsoft/TRELLIS) data processing instructions to preprocess the dataset. Make sure to follow their directory structure for compatibility and fully generate the necessary files and ``metadata.csv``.
3. Run the ``dataset_toolkits/build_metadata_scene.py`` script to create the scene-level metadata file:
    ```sh
    python dataset_toolkits/build_metadata_scene.py 3D-FUTURE 
    --output_dir <path_to_3D-FUTURE> 
    --set <train or test> 
    --vggt_ckpt checkpoints/VGGT-1B --save_mask
    ```
    This will generate a `metadata_scene.csv` file or a `metadata_scene_test.csv` file in the specified dataset directory.
4. For evaluation, run the ``dataset_toolkits/build_scene.sh`` script to render scene image for each scene(with Blender installed and the configs in the script set correctly):
    ```sh
    bash dataset_toolkits/build_scene.sh
    ```
    This will create a `scene_test_render` folder in the dataset directory containing the rendered images of the test scenes with Blender, which will be further used for evaluation.
## 🏋️‍♂️ Training
With the processed 3D-FUTURE dataset and the pretrained `ss_flow_img_dit_L_16l8_fp16.safetensors` model checkpoint from [TRELLIS](https://huggingface.co/microsoft/TRELLIS-image-large) correctly placed in the `checkpoints/scenegen/ckpts` directory, you can train SceneGen using the following command:
```
bash scripts/train.sh
```
For detailed training configurations, please refer to `configs/generation/ss_scenegen_flow_img_train.json` and change the parameters as needed.

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
- [x] Release Training Code
- [x] Release Data Processing Code
- [ ] Release Evaluation Code

## Acknowledgements
Many thanks to the code bases from [TRELLIS](https://github.com/microsoft/TRELLIS), [DINOv2](https://github.com/facebookresearch/dinov2), and [VGGT](https://github.com/facebookresearch/vggt).

## Contact
If you have any questions, please feel free to contact [meng-mou-xu@sjtu.edu.cn](mailto:meng-mou-xu@sjtu.edu.cn) and [haoningwu3639@gmail.com](mailto:haoningwu3639@gmail.com).
