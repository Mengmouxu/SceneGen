import os
os.environ['ATTN_BACKEND'] = 'xformers'     # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
from PIL import Image
from scenegen.pipelines import SceneGenImageToScenePipeline
import torch
import argparse
from easydict import EasyDict as edict
import sys
import numpy as np
import gradio
from gradio_litmodel3d import LitModel3D
from tqdm import tqdm
import contextlib
import io
import logging
import time

if __name__ == "__main__":
    dataset_name = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',type=str, required=True, help='Directory to save the metadata')
    parser.add_argument('--set', type=str, default='test', help='Test set only')
    parser.add_argument('--model_name', type=str, default='SceneGen', help='Model name to use for evaluation')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use for evaluation')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for evaluation')
    parser.add_argument('--gradio_only', action='store_true', help='Run Gradio interface only without evaluation')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    
    if dataset_name == '3D-FUTURE':
        assert opt.set == 'test', "3D-FUTURE dataset only supports test set"

        test_image_dir = os.path.join(opt.output_dir, f'masked_images_{opt.set}')
        assert os.path.exists(test_image_dir), f"Test image directory {test_image_dir} does not exist"

        scene_output_dir = os.path.join(opt.output_dir, f'scene_{opt.set}_{opt.model_name}_ablation_vggt_scene_mask')
        os.makedirs(scene_output_dir, exist_ok=True)

        scene_ids = os.listdir(test_image_dir)
        scene_ids = sorted(scene_ids)
        # scene_ids = scene_ids[:100]  # Limit to 100 scenes for testing
        if not opt.gradio_only:
            existing_ids = {f.split('.')[0] for f in os.listdir(scene_output_dir) if f.endswith('.glb')}
            scene_ids = [sid for sid in scene_ids if sid not in existing_ids]

        if opt.gpu_num > 1:
            scene_ids = scene_ids[opt.gpu_id::opt.gpu_num]

        if not opt.gradio_only:
            if opt.model_name == 'SceneGen':
                pipeline = SceneGenImageToScenePipeline.from_pretrained("../TRELLIS-image-large")
                pipeline.cuda()
            total_time = 0
            total_assets = 0
            for scene_id in tqdm(scene_ids, desc=f"Processing scenes on GPU {opt.gpu_id}", position=opt.gpu_id, leave=True):
                try:
                    Scene_path = os.path.join(test_image_dir, scene_id)
                    images_path = [
                        image_name for image_name in os.listdir(Scene_path)
                        if image_name.endswith(".png") and image_name != "scene.jpg" and "mask" not in image_name
                    ]
                    images_path = sorted(images_path) 
                    images = [
                        Image.open(os.path.join(Scene_path, image_name))
                        for image_name in images_path
                    ] 
                    mask_images_path = [
                        image_name for image_name in os.listdir(Scene_path)
                        if image_name.endswith(".png") and image_name != "scene.jpg" and "mask" in image_name and "masked_scene" not in image_name
                    ]

                    mask_images_path = sorted(mask_images_path)
                    mask_images = [
                        Image.open(os.path.join(Scene_path, image_name))
                        for image_name in mask_images_path
                    ]

                    if not os.path.exists(os.path.join(Scene_path, "scene.jpg")):
                        print(f"Scene image not found in {Scene_path}, skipping...")
                        continue

                    scene_image = Image.open(os.path.join(Scene_path, "scene.jpg"))

                    num_assets = len(mask_images_path)
                    if num_assets == 0:
                        print(f"No mask images found in {Scene_path}")
                        continue
                    if len(images) == 0:
                        print(f"No images found in {Scene_path}")
                        continue

                    if opt.model_name == 'SceneGen':
                        # Sort mask_images and images by mask size

                        # Calculate the size (number of white pixels) of each mask
                        mask_sizes = []
                        for i, mask in enumerate(mask_images):
                            # Convert mask to numpy array and count non-zero (white) pixels
                            mask_array = np.array(mask)
                            size = np.count_nonzero(mask_array)
                            mask_sizes.append((i, size))

                        query_asset_order = "largest"

                        # Sort indices by mask size (descending order - largest first)
                        sorted_indices = [idx for idx, _ in sorted(mask_sizes, key=lambda x: x[1], reverse=True)]
                        if query_asset_order != "largest":
                            sorted_indices = sorted_indices[::-1]
                        # Reorder the images and mask_images according to mask size
                        mask_images = [mask_images[i] for i in sorted_indices]
                        images = [images[i] for i in sorted_indices]

                        # Compute the inverse permutation to restore the original order of images
                        restore_indices = sorted(range(len(sorted_indices)), key=lambda i: sorted_indices[i])

                        # Redirect stdout and stderr to prevent disrupting tqdm progress bar
                        # Redirect stdout, stderr and suppress logging messages
                        original_log_level = logging.root.level
                        logging.root.setLevel(logging.ERROR)  # Suppress INFO and WARNING messages
                        
                        start_time = time.time()
                        with contextlib.redirect_stdout(io.StringIO()), \
                            contextlib.redirect_stderr(io.StringIO()):
                            outputs = pipeline.run_scene(
                                image=images,
                                mask_image=mask_images,
                                scene_image=scene_image,
                                preprocess_image=True,
                                sparse_structure_sampler_params={
                                        "steps": 25,
                                        "cfg_strength": 5.0,
                                        "cfg_interval": [0.5, 1.0],
                                        "rescale_t": 3.0
                                    },
                                slat_sampler_params={
                                        "steps": 25,
                                        "cfg_strength": 5.0,
                                        "cfg_interval": [0.5, 1.0],
                                        "rescale_t": 3.0
                                    },
                                resorted_indices=restore_indices,
                            )
                            torch.cuda.empty_cache()
                        end_time = time.time()
                        
                        # Restore original logging level
                        logging.root.setLevel(original_log_level)

                        scene = outputs["scene"]
                        scene.export(os.path.join(scene_output_dir, f"{scene_id}.glb"))

                        total_time += (end_time - start_time)
                        total_assets += num_assets
                except Exception as e:
                    print(f"Error processing scene {scene_id}: {e}")
                    continue
            
            if total_assets > 0:
                avg_time_per_asset = total_time / total_assets
                print(f"\nTotal assets processed: {total_assets}")
                print(f"Total generation time: {total_time:.2f} seconds")
                print(f"Average generation time per asset: {avg_time_per_asset:.2f} seconds")
        
        if opt.gradio_only:
            # Find common scene IDs between ground truth and generated scenes
            gt_scene_dir = os.path.join(opt.output_dir, f'scene_{opt.set}')
            gen_scene_dir = scene_output_dir  # This is already defined as opt.output_dir/f'scene_{opt.set}_{opt.model_name}'

            if os.path.exists(gt_scene_dir):
                gt_scene_ids = [f.split('.')[0] for f in os.listdir(gt_scene_dir) if f.endswith('.glb')]
                gen_scene_ids = [f.split('.')[0] for f in os.listdir(gen_scene_dir) if f.endswith('.glb')]
                
                # Find scene IDs that appear in both directories
                common_scene_ids = sorted(list(set(gt_scene_ids) & set(gen_scene_ids)))
                print(f"Found {len(common_scene_ids)} common scenes between ground truth and generated results")
            else:
                print(f"Ground truth scene directory {gt_scene_dir} does not exist")
                common_scene_ids = []

            if common_scene_ids:
                with gradio.Blocks() as demo:
                    gradio.Markdown("## 3D Scene Comparison Viewer")
                    
                    with gradio.Row():
                        scene_dropdown = gradio.Dropdown(
                            choices=common_scene_ids, 
                            label="Select Scene", 
                            value=common_scene_ids[0] if common_scene_ids else None
                        )
                        view_button = gradio.Button("View Scene")
                    
                    with gradio.Row():
                        with gradio.Column():
                            gradio.Markdown("### Ground Truth")
                            gt_model_output = LitModel3D(
                                label="Ground Truth Scene",
                                exposure=5.0,
                                height=500,
                                interactive=True,
                            )
                            gt_download_btn = gradio.DownloadButton(label="Download GT GLB", interactive=False)
                        
                        with gradio.Column():
                            gradio.Markdown(f"### Generated ({opt.model_name})")
                            gen_model_output = LitModel3D(
                                label="Generated Scene",
                                exposure=5.0,
                                height=500,
                                interactive=True,
                            )
                            gen_download_btn = gradio.DownloadButton(label="Download Generated GLB", interactive=False)
                    
                    def load_scenes(scene_id):
                        gt_path = os.path.join(gt_scene_dir, f"{scene_id}.glb")
                        gen_path = os.path.join(gen_scene_dir, f"{scene_id}.glb")
                        
                        gt_exists = os.path.exists(gt_path)
                        gen_exists = os.path.exists(gen_path)
                        
                        return (
                            gt_path if gt_exists else None,
                            gradio.update(interactive=gt_exists, value=gt_path if gt_exists else None),
                            gen_path if gen_exists else None,
                            gradio.update(interactive=gen_exists, value=gen_path if gen_exists else None),
                        )
                        
                    view_button.click(
                        load_scenes,
                        inputs=[scene_dropdown],
                        outputs=[gt_model_output, gt_download_btn, gen_model_output, gen_download_btn],
                    )
                        
                demo.launch(share=True)
                print("Gradio interface launched for scene comparison")
            else:
                print("No common scenes found for visualization")
        