import os
os.environ['ATTN_BACKEND'] = 'xformers'     # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.

import random
import tempfile
from typing import *
import gradio as gr
import numpy as np
import torch
from gradio_image_prompter import ImagePrompter
from gradio_litmodel3d import LitModel3D
from huggingface_hub import snapshot_download
from PIL import Image
from scenegen.pipelines import SceneGenImageToScenePipeline
from scenegen.utils.grounding_sam import plot_segmentation, segment
from scenegen.utils.inference_scene import run_scene
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Model paths
sam2_checkpoint = "./checkpoints/sam2-hiera-large/sam2_hiera_large.pt"
sam2_model_cfg = "configs/sam2/sam2_hiera_l.yaml"
scenegen_checkpoint = "checkpoints/scenegen"

sam2_predictor = SAM2ImagePredictor(
    build_sam2(sam2_model_cfg, sam2_checkpoint),
)
pipeline = SceneGenImageToScenePipeline.from_pretrained(scenegen_checkpoint)
pipeline.cuda()

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join("tmp")
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(TMP_DIR, exist_ok=True)

@torch.no_grad()
def run_segmentation(image_prompts: Any, polygon_refinement: bool, image_collection, seg_collection) -> tuple:
    rgb_image = image_prompts["image"].convert("RGB")

    if len(image_prompts["points"]) == 0:
        gr.Error("No points provided for segmentation. Please add points to the image.")
        return None, image_collection, seg_collection
    
    boxes = [
        [
            [int(box[0]), int(box[1]), int(box[3]), int(box[4])]
            for box in image_prompts["points"]
        ]
    ]

    detections = segment(
        sam2_predictor,
        rgb_image,
        boxes=[boxes],
        polygon_refinement=polygon_refinement,
    )
    seg_map_pil = plot_segmentation(rgb_image, detections)
    
    torch.cuda.empty_cache()

    return seg_map_pil, image_collection, seg_collection

def add_to_cache(rgb_image, seg_image, image_collection, seg_collection):
    if rgb_image is None or seg_image is None:
        gr.Warning("No image or segmentation to add to cache.")
        return image_collection, seg_collection, None
    
    new_image_collection = image_collection.copy()
    new_seg_collection = seg_collection.copy()
    
    new_image_collection.append(rgb_image)
    new_seg_collection.append(seg_image)
    
    preview_images = create_preview_images(new_image_collection, new_seg_collection)
    
    return new_image_collection, new_seg_collection, preview_images

def create_preview_images(image_collection, seg_collection):
    concat_image_collection = []
    for i in range(len(image_collection)):
        img_rgb = image_collection[i]
        img_seg = seg_collection[i]

        if img_rgb.height != img_seg.height:
            aspect_ratio = img_seg.width / float(img_seg.height)
            new_width = int(aspect_ratio * img_rgb.height)
            img_seg_resized = img_seg.resize((new_width, img_rgb.height), Image.Resampling.LANCZOS)
        else:
            img_seg_resized = img_seg
        
        dst_width = img_rgb.width + img_seg_resized.width
        dst_height = img_rgb.height
        
        concatenated_image = Image.new('RGB', (dst_width, dst_height))
        concatenated_image.paste(img_rgb, (0, 0))
        concatenated_image.paste(img_seg_resized, (img_rgb.width, 0))
        concat_image_collection.append(concatenated_image)

    preview_images = [(img, f"Pair {i+1}") for i, img in enumerate(concat_image_collection)]
    return preview_images

def delete_cached_image(idx, image_collection, seg_collection):
    if idx is None or idx >= len(image_collection) or idx < 0:
        gr.Warning("Invalid image selection.")
        return image_collection, seg_collection, None
    
    new_image_collection = image_collection.copy()
    new_seg_collection = seg_collection.copy()
    
    new_image_collection.pop(idx)
    new_seg_collection.pop(idx)
    
    if len(new_image_collection) == 0:
        return [], [], []
    
    preview_images = create_preview_images(new_image_collection, new_seg_collection)
    
    return new_image_collection, new_seg_collection, preview_images

def run_generation(
    rgb_image: Any,
    seg_image: Union[str, Image.Image],
    seed: int,
    randomize_seed: bool = False,
    ss_num_inference_steps: int = 50,
    ss_cfg_strength: float = 3.0,
    ss_cfg_interval_start: float = 0.0,
    ss_cfg_interval_end: float = 1.0,
    ss_rescale_t: float = 1.0,
    slat_num_inference_steps: int = 25,
    slat_cfg_strength: float = 5.0,
    slat_cfg_interval_start: float = 0.5,
    slat_cfg_interval_end: float = 1.0,
    slat_rescale_t: float = 3.0,
    asset_order: str = "order",
    positions_type: Literal['last', 'avg'] = 'last',
    simplify: float = 0.95,
    texture_size: int = 1024,
):
    # Construct intervals from individual values
    ss_cfg_interval = [ss_cfg_interval_start, ss_cfg_interval_end]
    slat_cfg_interval = [slat_cfg_interval_start, slat_cfg_interval_end]
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    if not isinstance(rgb_image, Image.Image) and "image" in rgb_image:
        rgb_image = rgb_image["image"]

    scene = run_scene(
        pipeline,
        rgb_image,
        seg_image,
        seed=seed,
        ss_num_inference_steps= ss_num_inference_steps,
        ss_cfg_strength=ss_cfg_strength,
        ss_cfg_interval=ss_cfg_interval,
        ss_rescale_t=ss_rescale_t,
        slat_num_inference_steps=slat_num_inference_steps,
        slat_cfg_strength=slat_cfg_strength,
        slat_cfg_interval=slat_cfg_interval,
        slat_rescale_t=slat_rescale_t,
        order=asset_order,
        positions_type=positions_type,
        simplify=simplify,
        texture_size=texture_size,
    )

    _, tmp_path = tempfile.mkstemp(suffix=".glb", prefix="scenegen_", dir=TMP_DIR)
    scene.export(tmp_path)

    torch.cuda.empty_cache()

    return tmp_path, tmp_path, seed

def clear_cache():
    return [], [], []

def load_example(scene_data, mask_path):
    if isinstance(scene_data, dict) and "image" in scene_data:
        img_rgb = Image.open(scene_data["image"]).convert("RGB")
    else:
        img_rgb = Image.open(scene_data).convert("RGB")
    img_seg = Image.open(mask_path).convert("RGB")
    
    image_prompts_value = {"image": img_rgb, "points": []}
    
    print("Example loaded successfully!")  # 这个应该会打印
    
    return image_prompts_value, img_seg, img_rgb

# Demo
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'> SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass </h1>")
    image_collection = gr.State([])
    seg_collection = gr.State([])

    current_image = gr.State(None)
    selected_image_idx = gr.State(None)

    # Inject CSS to change Generation Controls border to a lighter blue
    gr.HTML("""
    <style>
      /* Container styling for generation controls */
      #generation_controls {
        border: 1px solid #3b82f6 !important; /* blue border */
        background: transparent !important;
        box-shadow: none !important;
        border-radius: 8px;
        padding: 6px !important;
      }

      /* Light blue header for the accordion summary in generation controls */
      #generation_controls .gr-accordion-summary {
        background-color: rgba(59,130,246,0.06) !important;
        border-radius: 6px;
        padding: 6px 8px !important;
      }

      /* Ensure inner panels are transparent (no gray fills) */
      #generation_controls .gr-box,
      #generation_controls .gr-group,
      #generation_controls .gr-accordion-details {
        background: transparent !important;
      }

      /* Remove any default gray borders from inner elements */
      #generation_controls .gr-box,
      #generation_controls .gr-group {
        border-color: transparent !important;
      }

      /* Make all block markdown have black background and white text */
      .gr-markdown,
      .gr-markdown h1,
      .gr-markdown h2,
      .gr-markdown h3,
      .gr-markdown h4,
      .gr-markdown p,
      .gr-markdown li,
      .gr-markdown blockquote,
      .gr-examples .gr-markdown,
      .gr-gallery + .gr-markdown,
      .gr-gallery ~ .gr-markdown {
        background-color: #000 !important;
        color: #fff !important;
        padding: 6px 8px !important;
        border-radius: 6px;
        display: inline-block;
      }

      /* Tab headers and tab labels: subtle light-blue tint */
      .gr-tabs .gr-tabs-header,
      .gr-tabs .gr-tab,
      .gr-tabs .gr-tab button {
        background-color: rgba(59,130,246,0.04) !important;
        border-radius: 6px;
      }

      /* Accordion headers elsewhere */
      .gr-accordion .gr-accordion-summary {
        background-color: rgba(59,130,246,0.06) !important;
        border-radius: 6px;
        padding: 6px 8px !important;
      }

      /* Examples / gallery titles: also ensure readable on black background */
      .gr-examples .gr-markdown,
      .gr-gallery + .gr-markdown,
      .gr-gallery ~ .gr-markdown {
        background-color: #000 !important;
        color: #fff !important;
        padding: 4px 8px !important;
        border-radius: 6px;
        display: inline-block;
      }
    </style>
    """)
    with gr.Group():
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                    color: #2d3748; 
                    padding: 20px; 
                    border-radius: 12px; 
                    margin: 10px 0; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    border: 1px solid #cbd5e0;">
            <h2 style="margin-top: 0; color: #2d3748; text-align: center; font-weight: bold;">
                🚀 Quick Start Guide
            </h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px;">
                
                <div style="background: #ffffff; 
                            padding: 15px; 
                            border-radius: 8px; 
                            border-left: 4px solid #e53e3e;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: #e53e3e; margin-top: 0; font-weight: bold;">
                        📷 Step 1: Input & Segment
                    </h3>
                    <p style="margin: 5px 0; line-height: 1.5; color: #2d3748; font-weight: 500;">
                        <strong>1.</strong> Upload your scene image<br>
                        <strong>2.</strong> Use mouse to draw bounding boxes around objects<br>
                        <strong>3.</strong> Click <strong style="color: #e53e3e;">"Run Segmentation"</strong> to segment objects<br>
                        <em style="color: #718096;">※ For multi-image generation: maintain consistent object annotation order across all images</em>
                    </p>
                </div>
                
                <div style="background: #ffffff; 
                            padding: 15px; 
                            border-radius: 8px; 
                            border-left: 4px solid #38b2ac;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: #38b2ac; margin-top: 0; font-weight: bold;">
                        🗃️ Step 2: Manage Cache
                    </h3>
                    <p style="margin: 5px 0; line-height: 1.5; color: #2d3748; font-weight: 500;">
                        <strong>1.</strong> Click <strong style="color: #38b2ac;">"Add to Cache"</strong> when satisfied with segmentation<br>
                        <strong>2.</strong> Repeat Step 1-2 for multiple images<br>
                        <strong>3.</strong> Use <strong style="color: #38b2ac;">"Delete Selected"</strong> or <strong style="color: #38b2ac;">"Clear All"</strong> to manage cached images
                    </p>
                </div>
                
                <div style="background: #ffffff; 
                            padding: 15px; 
                            border-radius: 8px; 
                            border-left: 4px solid #3182ce;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: #3182ce; margin-top: 0; font-weight: bold;">
                        🎮 Step 3: Generate Scene
                    </h3>
                    <p style="margin: 5px 0; line-height: 1.5; color: #2d3748; font-weight: 500;">
                        <strong>1.</strong> Adjust generation parameters (optional)<br>
                        <strong>2.</strong> Click <strong style="color: #3182ce;">"Generate 3D Scene"</strong><br>
                        <strong>3.</strong> Download the generated GLB file when ready
                    </p>
                </div>
                
            </div>
            
            <div style="background: #fff3cd; 
                        padding: 12px; 
                        border-radius: 8px; 
                        margin-top: 15px; 
                        text-align: center;
                        border: 1px solid #ffeaa7;">
                <p style="margin: 0; color: #856404; font-weight: bold; font-size: 16px;">
                    💡 Pro Tip: Try the examples below to get started quickly!
                </p>
            </div>
        </div>
        """)

    def action_add_new_image():
        cleared_image_prompts = None
        cleared_seg_image = None
        return cleared_image_prompts, cleared_seg_image, None

    def select_cached_image(evt: gr.SelectData):
        return evt.index

    with gr.Row(equal_height=False):
        # Left column - Input and Segmentation
        with gr.Column(scale=1):
            with gr.Group():
                # gr.Markdown("### 📷 Input & Segmentation")
                with gr.Row():
                    image_prompts = ImagePrompter(label="Input Image", type="pil")
                    seg_image = gr.Image(
                        label="Segmentation Result", type="pil", format="png"
                    )
                
                with gr.Row():
                    add_new_image_button = gr.Button("➕ Add New Image", variant="secondary")
                    seg_button = gr.Button("🎯 Run Segmentation", variant="primary")
                
                with gr.Accordion("🔧 Segmentation Settings", open=False):
                    polygon_refinement = gr.Checkbox(label="Polygon Refinement", value=False)
            
            with gr.Group():
                # gr.Markdown("### 🗃️ Image Collection")
                with gr.Row():
                    cached_images_gallery = gr.Gallery(
                        label="Cached Images",
                        show_label=False,
                        elem_id="cached_images",
                        columns=3,
                        height=250,
                        object_fit="contain",
                        preview=True,
                    )
                with gr.Row():
                    add_to_cache_button = gr.Button("➕ Add to Cache", variant="secondary")
                    delete_selected_button = gr.Button("🗑️ Delete Selected", variant="secondary")
                    clear_cache_button = gr.Button("🧹 Clear All", variant="secondary")

        # Right column - Generation and Output
        with gr.Column(scale=1):
            # Use an Accordion (collapsed by default) and assign an id for styling
            with gr.Accordion("🎮 Generation Controls", open=False, elem_id="generation_controls"):
                with gr.Group():
                    with gr.Row():
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=0,
                        )
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    with gr.Tabs():
                        with gr.TabItem("Basic Settings"):
                            asset_order = gr.Radio(
                                label="Asset Order",
                                choices=["largest", "smallest", "order"],
                                value="largest",
                            )
                            positions_type = gr.Radio(
                                label="Positions Type",
                                choices=['last', 'avg'],
                                value='last',
                            )
                            simplify = gr.Slider(
                                label="Simplify",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.95,
                            )
                            texture_size = gr.Dropdown(
                                label="Texture Size",
                                choices=[512, 1024, 2048, 4096],
                                value=1024,
                                type="value",
                                allow_custom_value=False
                            )

                        with gr.TabItem("Structure Parameters"):
                            ss_num_inference_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=25,
                            )
                            ss_cfg_strength = gr.Slider(
                                label="CFG Scale",
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=5.0,
                            )
                            with gr.Row():
                                ss_cfg_interval_start = gr.Slider(
                                    label="CFG Interval Start",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=0.5,
                                )
                                ss_cfg_interval_end = gr.Slider(
                                    label="CFG Interval End",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=1.0,
                                )
                            ss_rescale_t = gr.Slider(
                                label="Rescale Factor",
                                minimum=0.0,
                                maximum=5.0,
                                step=0.1,
                                value=3.0,
                            )
                            
                        with gr.TabItem("SLAT Parameters"):
                            slat_num_inference_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=25,
                            )
                            slat_cfg_strength = gr.Slider(
                                label="CFG Scale",
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=5.0,
                            )
                            with gr.Row():
                                slat_cfg_interval_start = gr.Slider(
                                    label="CFG Interval Start",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=0.5,
                                )
                                slat_cfg_interval_end = gr.Slider(
                                    label="CFG Interval End",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=1.0,
                                )
                            slat_rescale_t = gr.Slider(
                                label="Rescale Factor",
                                minimum=0.0,
                                maximum=5.0,
                                step=0.1,
                                value=3.0,
                            )
                    
            gen_button = gr.Button("🚀 Generate 3D Scene", variant="primary", size="lg")

            with gr.Group():
                # gr.Markdown("### 🖼️ Generated Output")
                model_output = LitModel3D(
                    label="Generated 3D Scene",
                    exposure=5.0,
                    height=500,
                    interactive=True,
                )
                download_glb = gr.DownloadButton(label="📥 Download GLB File", variant="primary", interactive=True)

    # Click handler for the new "Add New Image" button
    add_new_image_button.click(
        fn=action_add_new_image,
        inputs=None,
        outputs=[image_prompts, seg_image, current_image]
    )

    seg_button.click(
        run_segmentation,
        inputs=[
            image_prompts,
            polygon_refinement,
            image_collection,
            seg_collection
        ],
        outputs=[
            seg_image,
            image_collection,
            seg_collection,
        ],
    ).then(
        lambda img_prompts: img_prompts["image"] if isinstance(img_prompts, dict) and "image" in img_prompts else None,
        inputs=[image_prompts],
        outputs=[current_image]
    ).then(lambda: gr.Button(interactive=True), outputs=[add_to_cache_button])

    add_to_cache_button.click(
        add_to_cache,
        inputs=[
            current_image,
            seg_image,
            image_collection,
            seg_collection
        ],
        outputs=[
            image_collection,
            seg_collection,
            cached_images_gallery
        ],
    ).then(lambda: gr.Button(interactive=True), outputs=[gen_button])

    cached_images_gallery.select(
        select_cached_image,
        inputs=None,
        outputs=[selected_image_idx]
    )

    delete_selected_button.click(
        delete_cached_image,
        inputs=[
            selected_image_idx,
            image_collection,
            seg_collection
        ],
        outputs=[
            image_collection,
            seg_collection,
            cached_images_gallery
        ],
    ).then(lambda: None, outputs=[selected_image_idx])

    clear_cache_button.click(
        clear_cache,
        inputs=[],
        outputs=[image_collection, seg_collection, cached_images_gallery]
    )

    gen_button.click(
        run_generation,
        inputs=[
            image_collection,
            seg_collection,
            seed,
            randomize_seed,
            ss_num_inference_steps,
            ss_cfg_strength,
            ss_cfg_interval_start,
            ss_cfg_interval_end,
            ss_rescale_t,
            slat_num_inference_steps,
            slat_cfg_strength,
            slat_cfg_interval_start,
            slat_cfg_interval_end,
            slat_rescale_t,
            asset_order,
            positions_type,
            simplify,
            texture_size,
        ],
        outputs=[model_output, download_glb, seed],
    ).then(lambda: gr.Button(interactive=True), outputs=[download_glb])

    # Load and display examples
    example_dir = "assets/gradio_demos"
    if os.path.exists(example_dir):
        example_folders = sorted([os.path.join(example_dir, d) for d in os.listdir(example_dir) if os.path.isdir(os.path.join(example_dir, d))])
        
        examples = []
        for folder in example_folders:
            scene_path = os.path.join(folder, "scene.jpg")
            mask_path = os.path.join(folder, "rgb_mask.png")
            if os.path.exists(scene_path) and os.path.exists(mask_path):
                examples.append([{"image": scene_path, "points": []}, mask_path])

        if examples:
            with gr.Group():
                with gr.Row():
                    example_images = gr.Examples(
                        examples=examples,
                        inputs=[image_prompts, seg_image],
                        outputs=[image_prompts, seg_image, current_image],
                        fn=load_example,
                        examples_per_page=7
                    )
                
                with gr.Row():
                    auto_process_button = gr.Button("🚀 Load Example & Auto Generate", variant="primary", size="lg")
                
            def auto_process_example(image_prompts, seg_image, image_collection, seg_collection,
                                   seed, randomize_seed, ss_num_inference_steps, ss_cfg_strength,
                                   ss_cfg_interval_start, ss_cfg_interval_end, ss_rescale_t,
                                   slat_num_inference_steps, slat_cfg_strength, slat_cfg_interval_start,
                                   slat_cfg_interval_end, slat_rescale_t, asset_order,
                                   positions_type, simplify, texture_size):
                
                print("Auto processing example...")
                
                if image_prompts is None or seg_image is None:
                    gr.Warning("Please select an example first!")
                    return (image_collection, seg_collection, None, None, None, None)
                
                rgb_image = image_prompts["image"] if isinstance(image_prompts, dict) and "image" in image_prompts else image_prompts
                
                new_image_collection = []
                new_seg_collection = []
                new_image_collection.append(rgb_image)
                new_seg_collection.append(seg_image)
                preview_images = create_preview_images(new_image_collection, new_seg_collection)
                
                print(f"Added to cache. Total images: {len(new_image_collection)}")
                
                model_path, download_path, new_seed = run_generation(
                    new_image_collection, new_seg_collection, seed, randomize_seed,
                    ss_num_inference_steps, ss_cfg_strength, ss_cfg_interval_start,
                    ss_cfg_interval_end, ss_rescale_t, slat_num_inference_steps,
                    slat_cfg_strength, slat_cfg_interval_start, slat_cfg_interval_end,
                    slat_rescale_t, asset_order, positions_type, simplify, texture_size
                )
                
                print("Generation completed!")
                
                return (new_image_collection, new_seg_collection, preview_images,
                        model_path, download_path, new_seed)
            
            auto_process_button.click(
                auto_process_example,
                inputs=[
                    image_prompts, seg_image, image_collection, seg_collection,
                    seed, randomize_seed, ss_num_inference_steps, ss_cfg_strength,
                    ss_cfg_interval_start, ss_cfg_interval_end, ss_rescale_t,
                    slat_num_inference_steps, slat_cfg_strength, slat_cfg_interval_start,
                    slat_cfg_interval_end, slat_rescale_t, asset_order,
                    positions_type, simplify, texture_size
                ],
                outputs=[
                    image_collection, seg_collection, cached_images_gallery,
                    model_output, download_glb, seed
                ]
            ).then(lambda: gr.Button(interactive=True), outputs=[download_glb])

demo.launch()