import os
import json
from pycocotools.coco import COCO
import trimesh
import sys
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm
import gradio as gr
from gradio_litmodel3d import LitModel3D
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import math

def transform_to_query(pos_target, euler_target, pos_query, euler_query):
    rot_query = R.from_euler('xyz', euler_query, degrees=True).as_matrix()
    rot_target = R.from_euler('xyz', euler_target, degrees=True).as_matrix()

    relative_pos = np.dot(rot_query.T, np.array(pos_target) - np.array(pos_query))

    relative_rot = np.dot(rot_query.T, rot_target)
    relative_quat = R.from_matrix(relative_rot).as_quat(scalar_first=True)

    return relative_pos, relative_quat

def transform_from_local(local_pos, local_euler, ref_pos, ref_euler):
    # Convert euler angles to rotation matrices
    if len(ref_euler) == 4:
        ref_euler = R.from_quat(ref_euler, scalar_first=True).as_euler('xyz', degrees=True)
    if len(local_euler) == 4:
        local_euler = R.from_quat(local_euler, scalar_first=True).as_euler('xyz', degrees=True)
        
    rot_ref = R.from_euler('xyz', ref_euler, degrees=True).as_matrix()
    local_rot = R.from_euler('xyz', local_euler, degrees=True).as_matrix()
    
    # Compute global position
    global_pos = np.dot(rot_ref, local_pos) + np.array(ref_pos)
    
    # Compute global rotation
    global_rot = np.dot(rot_ref, local_rot)
    global_euler = R.from_matrix(global_rot).as_quat(scalar_first=True)
    
    return tuple(global_pos), tuple(global_euler)

if __name__ == "__main__":
    dataset_name = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',type=str, required=True, help='Directory to save the metadata')
    parser.add_argument('--set', type=str, default='test', help='Test set only')
    parser.add_argument('--blender_path', type=str, default='/local/blender-3.0.1-linux-x64', help='Path to Blender directory')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use for evaluation')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for evaluation')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for parallel processing')
    parser.add_argument('--worker_id', type=int, default=0, help='Worker ID for parallel processing')
    parser.add_argument('--gradio_only', action='store_true', help='Run Gradio interface only without evaluation')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    
    if dataset_name == '3D-FUTURE':
        assert opt.set == 'test', "3D-FUTURE dataset only supports test set"
        os.makedirs(os.path.join(opt.output_dir, f"scene_{opt.set}"), exist_ok=True)

        scene_json_path = os.path.join(opt.output_dir, 'raw', '3D-FUTURE-scene', 'GT', f'{opt.set}_set.json')
        image_data_dir = os.path.join(opt.output_dir, 'raw', '3D-FUTURE-scene', f'{opt.set}', 'image')
        model_data_dir = os.path.join(opt.output_dir, 'raw', '3D-FUTURE-model')
        assert os.path.exists(scene_json_path), FileNotFoundError(f'{scene_json_path} not found, please check the dataset is correctly downloaded and placed under raw directory')
        assert os.path.exists(image_data_dir), FileNotFoundError(f'{image_data_dir} not found, please check the dataset is correctly downloaded and placed under raw directory')
        assert os.path.exists(model_data_dir), FileNotFoundError(f'{model_data_dir} not found, please check the dataset is correctly downloaded and placed under raw directory')

        print('Loading scene data...')
        scene_data = COCO(scene_json_path)
        scene_ids = scene_data.getImgIds()
        scene_images = scene_data.loadImgs(scene_ids)
        image_ids = [str(scene_image['id']) for scene_image in scene_images]
        if opt.gpu_num > 1:
            image_ids = image_ids[opt.gpu_id::opt.gpu_num]
            if opt.num_workers > 1:
                image_ids = image_ids[opt.worker_id::opt.num_workers]
        scene_render_cfg = {}

        for img_id in tqdm(image_ids, desc=f'Processing scenes on GPU {opt.gpu_id}', position=opt.gpu_id, leave=True):
            img_info = scene_data.loadImgs(int(img_id))[0]
            # Get the width and height of the image from img_info
            width = img_info['width']
            height = img_info['height']
            
            models = []
            translations = []
            eulers = []
            trimesh_scene = trimesh.Scene()
            camera_transform = np.eye(4)
            fov = math.radians(60)
            try:
                ann_ids = scene_data.getAnnIds(imgIds=int(img_id))
                annotations = scene_data.loadAnns(ann_ids)
                for ann in annotations:
                    model_id = ann['model_id']
                    if model_id is None:
                        continue
                    if 'pose' not in ann or ann['pose']['translation'] is None:
                        continue
                    model_path = os.path.join(model_data_dir, model_id, 'raw_model.obj')
                    if not os.path.exists(model_path):
                        continue
                    model = trimesh.load(model_path)
                    models.append(model)
                    translations.append(ann['pose']['translation'])
                    eulers.append(ann['pose']['euler'])
                    if 'fov' in ann:
                        fov = ann['fov']
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")
                continue
            if len(models) == 0:
                continue
            
            translations.append((0.0, 0.0, 0.0))
            eulers.append((0.0, 0.0, 0.0))

            quats = []
            poses = []
            for translation, euler in zip(translations, eulers):
                relative_pos, relative_quat = transform_to_query(translation, euler, translations[0], eulers[0])
                poses.append(relative_pos)
                quats.append(relative_quat)
            
            pos_query = (0.0, 0.0, 0.0)
            quat_query = R.from_euler('xyz', (90, 0, 0), degrees=True).as_quat(scalar_first=True)

            local_pos = []
            local_quat = []
            poses = poses[1:]  # Skip the first pose as it is the reference
            quats = quats[1:]  # Skip the first quaternion as it is the reference
            for pose, quat in zip(poses, quats):
                global_pos, global_quat = transform_from_local(pose, quat, pos_query, quat_query)
                local_pos.append(global_pos)
                local_quat.append(global_quat)
            positions = np.array([pos_query] + local_pos)
            quats = np.array([quat_query] + local_quat)

            for i in range(len(models)):
                model = models[i]
                transform = np.eye(4)
                rotation = R.from_quat(quats[i], scalar_first=True).as_matrix()
                transform[:3, :3] = rotation
                transform[:3, 3] = positions[i]
                trimesh_scene.add_geometry(model, transform=transform)

            # Normalize the scene to fit within (-1, -1, -1) and (1, 1, 1) with a margin.
            bounds = trimesh_scene.bounds
            scene_min, scene_max = bounds
            scene_center = (scene_min + scene_max) / 2.0
            extents = scene_max - scene_min
            max_extent = extents.max()

            # Define a margin (e.g., 2% margin from each side)
            margin = 0.02
            target_half_size = 1 - margin

            # Compute scale factor to fit the longest side of the scene into the target cube
            scale_factor = target_half_size * 2 / max_extent

            # Build transformation: first translate to origin, then scale.
            normalize_transform = trimesh.transformations.compose_matrix(
                translate=-scene_center,
                scale=[scale_factor, scale_factor, scale_factor]
            )

            trimesh_scene.apply_transform(normalize_transform)

            # Rotate the entire scene by -90 degrees around the x-axis
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle=np.radians(-90),
                direction=[1, 0, 0],
                point=[0, 0, 0]
            )
            trimesh_scene.apply_transform(rotation_matrix)

            trimesh_scene.export(os.path.join(opt.output_dir, f"scene_{opt.set}", f"{img_info['file_name']}.glb"))


            # Create a directory for rendered images if it doesn't exist
            render_dir = os.path.join(opt.output_dir, f"scene_{opt.set}_render")
            os.makedirs(render_dir, exist_ok=True)

            render_dir = os.path.join(opt.output_dir, f"scene_{opt.set}_render")
            os.makedirs(render_dir, exist_ok=True)
            
            render_path = os.path.join(render_dir, f"{img_info['file_name']}.png")
            
            camera_transform[:3, 3] = positions[-1]
            camera_transform[:3, :3] = R.from_quat(quats[-1], scalar_first=True).as_matrix()
            camera_transform = np.dot(normalize_transform, camera_transform)
            camera_pos = camera_transform[:3, 3].tolist()
            camera_rotation = R.from_matrix(camera_transform[:3, :3]).as_euler('xyz', degrees=True).tolist()
            
            scene_render_cfg[img_info['file_name']] = {
                'camera_pos': camera_pos,
                'camera_rotation': camera_rotation,
                'fov': fov,
                'render_path': render_path,
                'width': width,
                'height': height,
            }

            # Create a temporary Python script for Blender
            blender_script = os.path.join(opt.output_dir, ".cache", f"temp_render_script_{opt.gpu_id}_{opt.worker_id}.py")
            os.makedirs(os.path.dirname(blender_script), exist_ok=True)
            with open(blender_script, 'w') as f:
                f.write(f"""
import bpy
import math
import sys

# Clear existing objects
bpy.ops.wm.read_factory_settings(use_empty=True)
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

# Import the GLB file
bpy.ops.import_scene.gltf(filepath="{os.path.join(opt.output_dir, f"scene_{opt.set}", f"{img_info['file_name']}.glb")}")

# Set up rendering
scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# --- GPU Acceleration Setup ---
scene.cycles.device = 'GPU'
prefs = bpy.context.preferences
cycles_prefs = prefs.addons['cycles'].preferences

# Attempt to set compute device to OPTIX, fall back to CUDA, then HIP, then CPU
if hasattr(cycles_prefs, 'compute_device_type'):
    try:
        cycles_prefs.compute_device_type = 'OPTIX'
    except TypeError:
        try:
            cycles_prefs.compute_device_type = 'CUDA'
        except TypeError:
            try:
                cycles_prefs.compute_device_type = 'HIP'
            except TypeError:
                scene.cycles.device = 'CPU'

# Enable all available GPUs of the selected type
cycles_prefs.get_devices()
for device in cycles_prefs.devices:
    if device.type == cycles_prefs.compute_device_type:
        device.use = True

# Optimize render settings for speed
scene.cycles.samples = 128  # Lower samples for faster rendering
scene.cycles.use_denoising = True
if hasattr(scene.cycles, 'denoiser'):
    scene.cycles.denoiser = 'OPENIMAGEDENOISE' # Fast and good quality denoiser
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 2
scene.cycles.transparent_max_bounces = 2
scene.cycles.transmission_bounces = 2
scene.cycles.volume_bounces = 0

scene.render.resolution_x = {width}
scene.render.resolution_y = {height}
scene.render.resolution_percentage = 100
scene.render.film_transparent = True # Set to True to render with a transparent background

# Set render output format to PNG with RGBA
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# Set up the camera
cam_data = bpy.data.cameras.new('Camera')
cam_obj = bpy.data.objects.new('Camera', cam_data)
bpy.context.scene.collection.objects.link(cam_obj)

# Set camera FOV
cam_data.angle = {fov}

# Position camera
cam_obj.location = {camera_pos}
cam_obj.rotation_mode = 'XYZ'
cam_obj.rotation_euler = [math.radians(d) for d in {camera_rotation}]

# Set the camera as the active camera
bpy.context.scene.camera = cam_obj

# Set up lighting from multiple directions for full illumination
def add_light(name, location, energy, light_type='POINT'):
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_data.energy = energy
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location

# Add multiple point lights from different positions to ensure even lighting
add_light("Point_Front_Top_Right", (4, -4, 4), 2000)
add_light("Point_Front_Top_Left", (-4, -4, 4), 2000)
add_light("Point_Back_Top_Right", (4, 4, 4), 1500)
add_light("Point_Back_Top_Left", (-4, 4, 4), 1500)
add_light("Point_Bottom", (0, 0, -5), 1000)


# Render
scene.render.filepath = "{render_path}"
bpy.ops.render.render(write_still=True)
""")
            
            # Call Blender to render the scene
            blender_executable = os.path.join(opt.blender_path, "blender") if not opt.blender_path.endswith("blender") else opt.blender_path
            blender_cmd = f"{blender_executable} --background --python {blender_script} > /dev/null 2>&1"
            os.system(blender_cmd)
            os.remove(blender_script)
        
        # Save the scene render configuration to a JSON file
        render_cfg_path = os.path.join(opt.output_dir, f"scene_{opt.set}_render", "render_cfg.json")
        # If the config file exists, read it and update it. Otherwise, create a new one.
        # This is to merge results from multiple parallel processes.
        if os.path.exists(render_cfg_path):
            try:
                with open(render_cfg_path, 'r') as f:
                    existing_cfg = json.load(f)
                existing_cfg.update(scene_render_cfg)
                scene_render_cfg = existing_cfg
            except (json.JSONDecodeError, IOError):
                pass
        
        with open(render_cfg_path, 'w') as f:
            json.dump(scene_render_cfg, f, indent=4)
        
        if opt.gpu_id == 0 and opt.worker_id == 0:
            tqdm.write(f"3D-FUTURE scenes processed and saved to {opt.output_dir}/scene_{opt.set}")

    if opt.gradio_only:
        # Create a Gradio interface for viewing the 3D scenes
        # Create a file name list for the dropdown
        file_names = []
        for img_id in image_ids:
            img_info = scene_data.loadImgs(int(img_id))[0]
            file_names.append(img_info['file_name'])
        
        with gr.Blocks() as demo:
            gr.Markdown("## 3D-FUTURE Scene Viewer")
            
            with gr.Row():
                with gr.Column():
                    # Scene selection using file names
                    scene_dropdown = gr.Dropdown(
                        choices=file_names, 
                        label="Select Scene", 
                        value=file_names[0] if file_names else None
                    )
                    view_button = gr.Button("View Scene")
                
                with gr.Column():
                    # 3D model viewer
                    model_output = LitModel3D(
                        label="3D Scene",
                        exposure=5.0,
                        height=500,
                        interactive=True,
                    )
                    download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
            
            def load_scene(file_name):
                scene_path = os.path.join(opt.output_dir, f"scene_{opt.set}", f"{file_name}.glb")
                if os.path.exists(scene_path):
                    return scene_path, gr.update(interactive=True, value=scene_path)
                return None, gr.update(interactive=False)
                
            view_button.click(
                load_scene,
                inputs=[scene_dropdown],
                outputs=[model_output, download_glb],
            )
                
        demo.launch(share=True)
                
            
                    