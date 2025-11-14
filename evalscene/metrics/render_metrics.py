import os
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch
from torchvision import transforms
import lpips
import clip
from PIL import Image

@torch.no_grad()
def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val=1.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images using torch.

    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
        max_val (float): The maximum possible pixel value of the images.

    Returns:
        float: The PSNR value.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)

    return psnr.item()

@torch.no_grad()
def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, max_val=1.0):
    """
    Calculates the Structural Similarity Index (SSIM) between two images using torch.

    Args:
        img1 (torch.Tensor): The first image tensor (H, W, C) or (H, W).
        img2 (torch.Tensor): The second image tensor (H, W, C) or (H, W).
        max_val (float): The maximum possible pixel value of the images.

    Returns:
        float: The SSIM value.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    device = img1.device
    img1_t = img1.float()
    img2_t = img2.float()

    # Reshape to (N, C, H, W) format expected by torchmetrics
    if img1_t.ndim == 2:  # Grayscale image (H, W)
        img1_t = img1_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        img2_t = img2_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif img1_t.ndim == 3:  # Color image (C, H, W) or (H, W, C)
        # If image is (H, W, C), permute to (C, H, W)
        if img1_t.shape[2] <= 4: # A heuristic to check if the last dim is channel
            img1_t = img1_t.permute(2, 0, 1)
            img2_t = img2_t.permute(2, 0, 1)
        img1_t = img1_t.unsqueeze(0)  # (1, C, H, W)
        img2_t = img2_t.unsqueeze(0)  # (1, C, H, W)
    else:
        raise ValueError(f"Unsupported image dimension: {img1_t.ndim}")

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=max_val).to(device)
    
    return ssim_metric(img1_t, img2_t).item()

@torch.no_grad()
def compute_lpips(img1: torch.Tensor, img2: torch.Tensor, loss_fn, device='cuda'):
    """
    Calculates the Learned Perceptual Image Patch Similarity (LPIPS) between two images.

    Args:
        img1 (torch.Tensor): The first image tensor (C, H, W) in range [0, 1].
        img2 (torch.Tensor): The second image tensor (C, H, W) in range [0, 1].
        loss_fn (lpips.LPIPS): The pre-loaded LPIPS model.
        device (str): The device to run the computation on ('cpu' or 'cuda').

    Returns:
        float: The LPIPS distance.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # LPIPS model expects input tensors in the format (N, C, H, W) and normalized to the range [-1, 1].
    # Convert from (C, H, W) [0, 1] to (1, C, H, W) [-1, 1]
    def process_tensor(img_tensor):
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = (img_tensor * 2) - 1
        return img_tensor

    img1_t = process_tensor(img1).to(device)
    img2_t = process_tensor(img2).to(device)

    with torch.no_grad():
        dist = loss_fn.forward(img1_t, img2_t).item()

    return dist

def compute_fid(imgs1, imgs2, device='cuda', batch_size=16):
    """
    Calculates the Fréchet Inception Distance (FID) between two sets of images in batches to avoid OOM.

    Args:
        imgs1 (list[np.ndarray]): The first set of images, a list of NumPy arrays (H, W, C) in range [0, 255].
        imgs2 (list[np.ndarray]): The second set of images, a list of NumPy arrays (H, W, C) in range [0, 255].
        device (str): The device to run the computation on ('cpu' or 'cuda').
        batch_size (int): Number of images per batch.

    Returns:
        float: The FID score.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        raise ImportError("Please install torchmetrics: pip install torchmetrics")

    if not imgs1 or not imgs2:
        raise ValueError("Input image lists cannot be empty.")

    fid = FrechetInceptionDistance(feature=2048).to(device)

    def to_tensor(img):
        img_permuted = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img_permuted).to(torch.uint8)

    # Process imgs1 in batches
    for i in range(0, len(imgs1), batch_size):
        batch_imgs = imgs1[i:i+batch_size]
        batch_tensor = torch.stack([to_tensor(img) for img in batch_imgs]).to(device)
        fid.update(batch_tensor, real=True)
        del batch_tensor

    # Process imgs2 in batches
    for i in range(0, len(imgs2), batch_size):
        batch_imgs = imgs2[i:i+batch_size]
        batch_tensor = torch.stack([to_tensor(img) for img in batch_imgs]).to(device)
        fid.update(batch_tensor, real=False)
        del batch_tensor

    return fid.compute().item()

@torch.no_grad()
def compute_clip_similarity(img1, img2, model, preprocess, device='cuda'):
    """
    Calculates the CLIP similarity score between two images using a pre-loaded model.

    Args:
        img1 (Image.Image): The first image (PIL Image).
        img2 (Image.Image): The second image (PIL Image).
        model: The pre-loaded CLIP model.
        preprocess: The pre-loaded CLIP preprocess transform.
        device (str): The device to run the computation on ('cpu' or 'cuda').

    Returns:
        float: The CLIP similarity score.
    """

    # Preprocess images and move to the specified device
    img1_processed = preprocess(img1).unsqueeze(0).to(device)
    img2_processed = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode images to get their feature vectors
        img1_features = model.encode_image(img1_processed)
        img2_features = model.encode_image(img2_processed)

        # Normalize the features
        img1_features /= img1_features.norm(dim=-1, keepdim=True)
        img2_features /= img2_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity
        similarity = (img1_features @ img2_features.T).item()

    return similarity

@torch.no_grad()
def compute_dinov2_similarity(img1: Image.Image, img2: Image.Image, model, preprocess, device='cuda'):
    """
    Calculates the DINOv2 similarity score between two images using a pre-loaded model.

    Args:
        img1 (Image.Image): The first image (PIL Image).
        img2 (Image.Image): The second image (PIL Image).
        model: The pre-loaded DINOv2 model.
        preprocess: The pre-loaded DINOv2 preprocess transform.
        device (str): The device to run the computation on ('cpu' or 'cuda').

    Returns:
        float: The DINOv2 similarity score (cosine similarity).
    """

    # Preprocess images and move to the specified device
    img1_processed = preprocess(img1).unsqueeze(0).to(device)
    img2_processed = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode images to get their feature vectors
        img1_features = model(img1_processed)
        img2_features = model(img2_processed)

        # Normalize the features (L2 norm)
        img1_features /= img1_features.norm(dim=-1, keepdim=True)
        img2_features /= img2_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(img1_features, img2_features).item()

    return similarity

def compute_render_metrics_(img1, img2, loss_fn, clip_model, clip_preprocess, dinov2_model, dinov2_preprocess, device='cuda', name="render"):
    """
    Computes various render metrics between two images.

    Args:
        img1 (torch.Tensor): The first image tensor (C, H, W) normalized to [0, 1].
        img2 (torch.Tensor): The second image tensor (C, H, W) normalized to [0, 1].
        loss_fn: The pre-loaded LPIPS model.
        clip_model: The pre-loaded CLIP model.
        clip_preprocess: The pre-loaded CLIP preprocess function.
        dinov2_model: The pre-loaded DINOv2 model.
        dinov2_preprocess: The pre-loaded DINOv2 preprocess function.
        device (str): The device to run the computation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Ensure the input tensors are on the correct device
    img1 = img1.to(device)
    img2 = img2.to(device)
    # Compute PSNR, SSIM, and LPIPS
    psnr = compute_psnr(img1, img2)
    ssim = compute_ssim(img1, img2)
    lpips_value = compute_lpips(img1, img2, loss_fn, device)

    # Convert torch tensors to PIL images for CLIP and DINOv2
    to_pil = transforms.ToPILImage()
    img1_pil = to_pil(img1.cpu())
    img2_pil = to_pil(img2.cpu())

    # Compute CLIP and DINOv2 similarities
    clip_similarity = compute_clip_similarity(img1_pil, img2_pil, clip_model, clip_preprocess, device)
    dinov2_similarity = compute_dinov2_similarity(img1_pil, img2_pil, dinov2_model, dinov2_preprocess, device)

    metrics = {
        f'{name}_psnr': np.array([psnr]),
        f'{name}_ssim': np.array([ssim]),
        f'{name}_lpips': np.array([lpips_value]),
        f'{name}_clip_similarity': np.array([clip_similarity]),
        f'{name}_dinov2_similarity': np.array([dinov2_similarity])
    }

    return metrics

def load_lpips_model(net='alex', device='cuda'):
    """
    Loads the LPIPS model.

    Args:
        net (str): The network backbone to use ('alex' or 'vgg').
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        lpips.LPIPS: The loaded LPIPS model.
    """
    loss_fn = lpips.LPIPS(net=net).to(device)
    return loss_fn

def load_clip_model(model_name="ViT-B/32", device='cuda'):
    """
    Loads the CLIP model and preprocess function.

    Args:
        model_name (str): The name of the CLIP model to load (e.g., "ViT-B/32").
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        model: The loaded CLIP model.
        preprocess: The preprocess function for CLIP.
    """
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess

def load_dinov2_model(model_name="facebook/dinov2-base", device='cuda'):
    """
    Loads the DINOv2 model and image processor using the transformers library.

    Args:
        model_name_or_path (str): The model identifier from the Hugging Face Hub
                                  or path to a local directory.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        model: The loaded DINOv2 model.
        processor: The image processor for DINOv2.
    """
    cache_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    
    if os.path.exists(cache_dir):
        model = torch.hub.load(cache_dir, model_name, source='local', pretrained=True)
    else:
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
    model.to(device)
    processor = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, processor
