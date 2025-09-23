import os
import shutil
import sys
import time
import importlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import cv2
from matplotlib.patches import Polygon
from easydict import EasyDict as edict
from concurrent.futures import ProcessPoolExecutor
import torch
import trimesh
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vggt'))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import random


if __name__ == '__main__':
    dataset_name = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',type=str, required=True, help='Directory to save the metadata')
    parser.add_argument('--set', type=str, default='train', help='Set to build metadata for')
    parser.add_argument('--vggt_ckpt', type=str, default='checkpoints/VGGT-1B', help='Path to the VGGT checkpoint')
    parser.add_argument('--statistic_only', action='store_true', default=False, help='Only generate statistics without processing images')
    parser.add_argument('--save_mask', action='store_true', default=False, help='Save maskes')
    parser.add_argument('--use_render_cond', action='store_true', default=False, help='Use renders_cond as additional data')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    timestamp = str(int(time.time()))
    
    if dataset_name == '3D-FUTURE':
        assert os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')), FileNotFoundError('metadata.csv not found in the output directory')
        print('Loading previous metadata...')
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
        metadata.set_index('sha256', inplace=True)
        
        if opt.set == 'train':
            if os.path.exists(os.path.join(opt.output_dir, 'metadata_scene.csv')):
                print('Loading previous metadata_scene...')
                metadata_scene = pd.read_csv(os.path.join(opt.output_dir, 'metadata_scene.csv'))
            else:
                metadata_scene = pd.DataFrame(columns=['scene_id', 'model_id', 'translation', 'rotation', 'scale', 'vggt_feature', 'masked_image_path', 'scene_image_path',
                                                    'sha256','file_identifier','aesthetic_score','captions','rendered',
                                                    'voxelized','num_voxels','cond_rendered','local_path',
                                                    'ss_latent_ss_enc_conv3d_16l8_fp16'])
        elif opt.set == 'test':
            if os.path.exists(os.path.join(opt.output_dir, 'metadata_scene_test.csv')):
                print('Loading previous metadata_scene_test...')
                metadata_scene = pd.read_csv(os.path.join(opt.output_dir, 'metadata_scene_test.csv'))
            else:
                metadata_scene = pd.DataFrame(columns=['scene_id', 'model_id', 'translation', 'rotation', 'scale', 'vggt_feature', 'masked_image_path', 'scene_image_path',
                                                    'sha256','file_identifier','aesthetic_score','captions','rendered',
                                                    'voxelized','num_voxels','cond_rendered','local_path',
                                                    'ss_latent_ss_enc_conv3d_16l8_fp16'])
        
        scene_json_path = os.path.join(opt.output_dir, 'raw', '3D-FUTURE-scene', 'GT', f'{opt.set}_set.json')
        image_data_dir = os.path.join(opt.output_dir, 'raw', '3D-FUTURE-scene', f'{opt.set}', 'image')
        model_data_dir = os.path.join(opt.output_dir, 'raw', '3D-FUTURE-model')
        renders_cond_dir = os.path.join(opt.output_dir, 'renders_cond')
        assert os.path.exists(scene_json_path), FileNotFoundError(f'{scene_json_path} not found, please check the dataset is correctly downloaded and placed under raw directory')
        assert os.path.exists(image_data_dir), FileNotFoundError(f'{image_data_dir} not found, please check the dataset is correctly downloaded and placed under raw directory')
        assert os.path.exists(model_data_dir), FileNotFoundError(f'{model_data_dir} not found, please check the dataset is correctly downloaded and placed under raw directory')
        assert os.path.exists(renders_cond_dir), FileNotFoundError(f'{renders_cond_dir} not found, please check the dataset is correctly downloaded and placed under raw directory')
        
        if opt.save_mask:
            print('Loading scene data...')
            scene_data = COCO(scene_json_path)
            scene_ids = scene_data.getImgIds()
            scene_images = scene_data.loadImgs(scene_ids)
            image_ids = [str(scene_image['id']) for scene_image in scene_images]
            image_ids = image_ids

            for img_id in tqdm(image_ids, desc='Processing masks', unit='mask'):
                try:
                    if str(img_id) in metadata_scene['scene_id'].astype(str).values:
                        continue
                    img_info = scene_data.loadImgs(int(img_id))[0]
                    img_path = os.path.join(image_data_dir, img_info['file_name']+".jpg")
                    if not os.path.exists(img_path):
                        continue
                    ann_ids = scene_data.getAnnIds(imgIds=int(img_id))
                    annotations = scene_data.loadAnns(ann_ids)
                    output_img_dir = os.path.join(opt.output_dir, 'masked_images', img_info['file_name'])
                    if opt.set == 'test':
                        output_img_dir = os.path.join(opt.output_dir, 'masked_images_test', img_info['file_name'])
                    os.makedirs(output_img_dir, exist_ok=True)
                    instance_id = 0

                    for ann in annotations:
                        model_id = ann['model_id']
                        if model_id is None:
                            continue
                        if 'pose' not in ann or ann['pose']['translation'] is None:
                            continue
                        model_path = os.path.join(model_data_dir, model_id, 'raw_model.obj')
                        if not os.path.exists(model_path):
                            # print(f"Model path does not exist: {model_path}")
                            continue
                        
                        if isinstance(ann['segmentation'], dict):
                            try:
                                mask = maskUtils.decode(ann['segmentation'])
                            except TypeError as e:
                                try:
                                    h, w = img_info['height'], img_info['width']
                                    rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                                    mask = maskUtils.decode(rle)
                                except Exception as e2:
                                    print(f"Failed to decode mask: {e2}")
                                    mask = np.zeros((h, w), dtype=np.uint8)
                        else:
                            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                            for seg in ann['segmentation']:
                                poly = np.array(seg).reshape((len(seg)//2, 2))
                                cv2.fillPoly(mask, [np.int32(poly)], 1)
                        
                        if mask is None or mask.sum() == 0:
                            continue

                        # Convert single-channel mask to three-channel image
                        mask_3channel = np.stack([(mask * 255).astype(np.uint8)] * 3, axis=2)
                        mask_output_path = os.path.join(output_img_dir, f"{instance_id}_mask.png")
                        cv2.imwrite(mask_output_path, mask_3channel)
                        instance_id += 1
                except Exception as e:
                    print(f"Error processing image {img_id}: {e}")
                    continue

        if not opt.statistic_only:
            print('Loading scene data...')
            scene_data = COCO(scene_json_path)
            print('Loading VGGT...')
            time_start = time.time()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            vggt_model = VGGT.from_pretrained(opt.vggt_ckpt).to(device)
            time_load_model = time.time() - time_start
            print(f'Done (t={time_load_model:.2f}s).')
            print('Processing scene data...')

            scene_ids = scene_data.getImgIds()
            scene_images = scene_data.loadImgs(scene_ids)
            image_ids = [str(scene_image['id']) for scene_image in scene_images]
            image_ids = image_ids

            for img_id in tqdm(image_ids, desc='Processing images', unit='image'):
                try:
                    if str(img_id) in metadata_scene['scene_id'].astype(str).values:
                        continue
                    img_info = scene_data.loadImgs(int(img_id))[0]
                    img_path = os.path.join(image_data_dir, img_info['file_name']+".jpg")
                    if not os.path.exists(img_path):
                        continue
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    ann_ids = scene_data.getAnnIds(imgIds=int(img_id))
                    annotations = scene_data.loadAnns(ann_ids)
                    
                    output_img_dir = os.path.join(opt.output_dir, 'masked_images', img_info['file_name'])
                    if opt.set == 'test':
                        output_img_dir = os.path.join(opt.output_dir, 'masked_images_test', img_info['file_name'])
                    os.makedirs(output_img_dir, exist_ok=True)

                    cv2.imwrite(os.path.join(output_img_dir, 'scene.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    instance_id = 0

                    vggt_image_input = load_and_preprocess_images([img_path]).to(device)
                    with torch.no_grad():
                        with torch.amp.autocast(dtype=dtype, device_type=device):
                            vggt_image_input = vggt_image_input[None]
                            prediction, _ = vggt_model.aggregator(vggt_image_input)
                            vggt_feature = prediction[-1][0, ...].cpu().numpy()
                            npz_save_dir = os.path.join(opt.output_dir, 'vggt_features')
                            if opt.set == 'test':
                                npz_save_dir = os.path.join(opt.output_dir, 'vggt_features_test')
                            os.makedirs(npz_save_dir, exist_ok=True)
                            npz_save_path = os.path.join(npz_save_dir, f"{img_info['file_name']}.npz")
                            np.savez_compressed(npz_save_path, vggt_feature=vggt_feature)

                    for ann in annotations:
                        model_id = ann['model_id']
                        if model_id is None:
                            continue
                        if 'pose' not in ann or ann['pose']['translation'] is None:
                            continue
                        model_path = os.path.join(model_data_dir, model_id, 'raw_model.obj')
                        if not os.path.exists(model_path):
                            # print(f"Model path does not exist: {model_path}")
                            continue
                        
                        if isinstance(ann['segmentation'], dict):  # RLE format
                            # Try standard decoding first
                            try:
                                mask = maskUtils.decode(ann['segmentation'])
                            except TypeError as e:
                                # If that fails, try converting to the proper RLE format
                                try:
                                    h, w = img_info['height'], img_info['width']
                                    rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                                    mask = maskUtils.decode(rle)
                                except Exception as e2:
                                    print(f"Failed to decode mask: {e2}")
                                    # Create empty mask as fallback
                                    mask = np.zeros((h, w), dtype=np.uint8)
                        else:  # Polygon format
                            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                            for seg in ann['segmentation']:
                                poly = np.array(seg).reshape((len(seg)//2, 2))
                                cv2.fillPoly(mask, [np.int32(poly)], 1)

                        
                        # Crop the image using the mask and add transparency
                        if mask is None or mask.sum() == 0:
                            # If mask is empty, skip this annotation
                            continue

                        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                        cropped_img = img[y:y+h, x:x+w]
                        cropped_mask = mask[y:y+h, x:x+w]

                        # Add an alpha channel to the cropped image
                        cropped_img_with_alpha = np.dstack((cropped_img, (cropped_mask * 255).astype(np.uint8)))

                        # Resize the cropped image to have the longer side as 518, and pad the shorter side
                        long_side = 518
                        h, w, _ = cropped_img_with_alpha.shape
                        scale = long_side / max(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        resized_img = cv2.resize(cropped_img_with_alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                        # Calculate padding
                        pad_h = (long_side - new_h) // 2
                        pad_w = (long_side - new_w) // 2
                        padded_img = cv2.copyMakeBorder(
                            resized_img, pad_h, long_side - new_h - pad_h, pad_w, long_side - new_w - pad_w,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
                        )

                        # Save or process the padded image as needed
                        output_img_path = os.path.join(opt.output_dir, 'masked_images', img_info['file_name'], f"{instance_id}.png")
                        if opt.set == 'test':
                            output_img_path = os.path.join(opt.output_dir, 'masked_images_test', img_info['file_name'], f"{instance_id}.png")
                        instance_id += 1

                        cv2.imwrite(output_img_path, cv2.cvtColor(padded_img, cv2.COLOR_RGBA2BGRA))

                        model_translation = ann['pose']['translation']
                        model_rotation = ann['pose']['euler']

                        model_size = trimesh.load(model_path).bounding_box.extents
                        model_scale = max(model_size)
                        # Find sha256 from metadata where the ID after "file_identifier/" equals model_id
                        sha256_match = None
                        if 'file_identifier' in metadata.columns:
                            # Extract the id from file_identifier by taking the substring after the last '/'
                            matching = metadata[metadata['file_identifier'].apply(
                                lambda fid: fid.split('/')[-1] if isinstance(fid, str) and '/' in fid else None
                            ) == model_id]
                            if not matching.empty:
                                # Since metadata index is sha256, get the first matching sha256 value
                                sha256_match = matching.index[0]
                        if sha256_match is not None:
                            # Append sha256_match to the new metadata row
                            new_row = pd.DataFrame([{
                                'scene_id': img_info['file_name'],
                                'model_id': model_id,
                                'translation': model_translation,
                                'rotation': model_rotation,
                                'scale': model_scale,
                                'vggt_feature': f"vggt_features/{img_info['file_name']}.npz",
                                'masked_image_path': f"masked_images/{img_info['file_name']}/{instance_id-1}.png",
                                'scene_image_path': f"masked_images/{img_info['file_name']}/scene.jpg",
                                'sha256': sha256_match,
                                'file_identifier': metadata.loc[sha256_match, 'file_identifier'],
                                'aesthetic_score': metadata.loc[sha256_match, 'aesthetic_score'],
                                'captions': metadata.loc[sha256_match, 'captions'],
                                'rendered': metadata.loc[sha256_match, 'rendered'],
                                'voxelized': metadata.loc[sha256_match, 'voxelized'],
                                'num_voxels': metadata.loc[sha256_match, 'num_voxels'],
                                'cond_rendered': metadata.loc[sha256_match, 'cond_rendered'],
                                'local_path': metadata.loc[sha256_match, 'local_path'],
                                'ss_latent_ss_enc_conv3d_16l8_fp16': metadata.loc[sha256_match, 'ss_latent_ss_enc_conv3d_16l8_fp16']
                            }])
                            metadata_scene = pd.concat([metadata_scene, new_row], ignore_index=True)
                except Exception as e:
                    print(f"Error processing image {img_id}: {e}")
                    continue

            if os.path.exists(renders_cond_dir) and opt.set == 'train' and opt.use_render_cond:
                print('Processing renders_cond data...')
                model_renders_cond_dir = os.listdir(renders_cond_dir)
                for model_dir in tqdm(model_renders_cond_dir, desc='Processing model renders_cond', unit='model'):
                    try:
                        if model_dir in metadata_scene['scene_id'].values:
                            continue
                        model_dir_path = os.path.join(renders_cond_dir, model_dir)
                        model_sha256 = model_dir
                        png_files = [f for f in os.listdir(model_dir_path) if f.lower().endswith('.png')]
                        if not png_files:
                            continue
                        model_image_path = os.path.join(model_dir_path, random.choice(png_files))
                        vggt_image_input = load_and_preprocess_images([model_image_path]).to(device)
                        model_id = metadata['file_identifier'][model_sha256].split('/')[-1]
                        with torch.no_grad():
                            with torch.amp.autocast(dtype=dtype, device_type=device):
                                vggt_image_input = vggt_image_input[None]
                                prediction, _ = vggt_model.aggregator(vggt_image_input)
                                vggt_feature = prediction[-1][0, ...].cpu().numpy()
                                npz_save_dir = os.path.join(opt.output_dir, 'vggt_features')
                                os.makedirs(npz_save_dir, exist_ok=True)
                                npz_save_path = os.path.join(npz_save_dir, f"{model_sha256}.npz")
                                np.savez_compressed(npz_save_path, vggt_feature=vggt_feature)
                        if model_id is None:
                            continue
                        new_row = pd.DataFrame([{
                            'scene_id': model_sha256,
                            'model_id': model_id,
                            'translation': False,
                            'rotation': False,
                            'scale': False,
                            'vggt_feature': f"vggt_features/{model_sha256}.npz",
                            'masked_image_path': f"renders_cond/{model_sha256}/{os.path.basename(model_image_path)}",
                            'scene_image_path': f"renders_cond/{model_sha256}/{os.path.basename(model_image_path)}",
                            'sha256': model_sha256,
                            'file_identifier': metadata['file_identifier'][model_sha256],
                            'aesthetic_score': metadata['aesthetic_score'][model_sha256],
                            'captions': metadata['captions'][model_sha256],
                            'rendered': metadata['rendered'][model_sha256],
                            'voxelized': metadata['voxelized'][model_sha256],
                            'num_voxels': metadata['num_voxels'][model_sha256],
                            'cond_rendered': metadata['cond_rendered'][model_sha256],
                            'local_path': metadata['local_path'][model_sha256],
                            'ss_latent_ss_enc_conv3d_16l8_fp16': metadata['ss_latent_ss_enc_conv3d_16l8_fp16'][model_sha256]
                        }])
                        metadata_scene = pd.concat([metadata_scene, new_row], ignore_index=True)
                    except Exception as e:
                        print(f"Error processing model renders_cond {model_dir}: {e}")
                        continue
            metadata_scene['scene_id'] = metadata_scene['scene_id'].astype(str)
            if opt.set == 'train':
                metadata_scene.to_csv(os.path.join(opt.output_dir, 'metadata_scene.csv'), index=False)
            else:
                metadata_scene.to_csv(os.path.join(opt.output_dir, 'metadata_scene_test.csv'), index=False)

            num_scene_downloaded = metadata_scene['local_path'].count() if 'local_path' in metadata_scene.columns else 0
        else:
            print('Statistics only mode, skipping image processing.')

        # Generate statistics
        statistic_file = os.path.join(opt.output_dir, 'statistics_scene.txt')
        if opt.set == 'test':
            statistic_file = os.path.join(opt.output_dir, 'statistics_scene_test.txt')
        with open(statistic_file, 'w') as f:
            f.write('Scene Statistics:\n')

            scene_id_series = metadata_scene['scene_id'].astype(str)
            mask_scene = scene_id_series.str.len() == 7
            mask_asset = ~mask_scene

            count_scene = len(set(metadata_scene.loc[mask_scene, 'scene_id']))
            count_asset = mask_asset.sum()

            models_in_scene_count_scene = metadata_scene.loc[mask_scene, 'model_id'].nunique()
            models_in_scene_count_asset = metadata_scene.loc[mask_asset, 'model_id'].nunique()

            masked_images_count_scene = metadata_scene.loc[mask_scene, 'masked_image_path'].count()
            masked_images_count_asset = metadata_scene.loc[mask_asset, 'masked_image_path'].count()
            f.write(f'  - Number of scene: {count_scene}\n')
            f.write(f'    - Number of models in scene: {models_in_scene_count_scene}\n')
            f.write(f'    - Number of masked images in scene: {masked_images_count_scene}\n')
            # Count how many times each scene_id appears in mask_scene
            scene_occurrence = metadata_scene.loc[mask_scene, 'scene_id'].value_counts()
            # Summarize how many scenes appear X times
            occurrence_summary = scene_occurrence.value_counts().sort_index()
            f.write('    - Scene occurrence counts in mask_scene:\n')
            for occ, scene_count in occurrence_summary.items():
                f.write(f'        - {scene_count} scenes have {occ} instances{"s" if occ > 1 else ""}\n')
            
            f.write(f'  - Number of single asset scene: {count_asset}\n')
            f.write(f'    - Number of models in single asset: {models_in_scene_count_asset}\n')
            f.write(f'    - Number of masked images in single asset: {masked_images_count_asset}\n')
        with open(os.path.join(opt.output_dir, 'statistics.txt'), 'w') as f:
            for line in open(os.path.join(opt.output_dir, 'statistics_scene.txt')):
                if line.strip():  # Only print non-empty lines
                    print(line.rstrip())