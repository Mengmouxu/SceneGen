import torch
import numpy as np
import trimesh
from .chamfer_distance import ChamferDistance


def compute_chamfer_distance(
    pred: torch.Tensor, gt: torch.Tensor, chamfer_distance_cls: torch.nn.Module = None, chunk_size: int = 2048
):
    if chamfer_distance_cls is None:
        chamfer_distance_cls = ChamferDistance(chunk_size=chunk_size).to(pred.device)
    
    dist1, dist2 = chamfer_distance_cls(pred, gt)
    
    dist1 = dist1.mean(dim=1)
    dist2 = dist2.mean(dim=1)
    
    return dist1 + dist2, dist1, dist2


def compute_fscore(pred, gt, tau=0.1, chunk_size=2048):
    B, N, _ = pred.shape
    _, M, _ = gt.shape
    
    min_dists_pred_to_gt = torch.zeros(B, N, device=pred.device)
    min_dists_gt_to_pred = torch.zeros(B, M, device=gt.device)
    
    for b in range(B):
        pred_b = pred[b]
        gt_b = gt[b]
        
        for i in range(0, N, chunk_size):
            pred_chunk = pred_b[i : i + chunk_size]
            dists = torch.cdist(
                pred_chunk.unsqueeze(0), gt_b.unsqueeze(0), p=2
            )
            min_dists = dists.min(dim=2).values.squeeze(0)
            min_dists_pred_to_gt[b, i : i + min(chunk_size, N-i)] = min_dists[:min(chunk_size, N-i)]
        
        for i in range(0, M, chunk_size):
            gt_chunk = gt_b[i : i + chunk_size]
            dists = torch.cdist(
                gt_chunk.unsqueeze(0), pred_b.unsqueeze(0), p=2
            )
            min_dists = dists.min(dim=2).values.squeeze(0)
            min_dists_gt_to_pred[b, i : i + min(chunk_size, M-i)] = min_dists[:min(chunk_size, M-i)]
    
    precision_matches = (min_dists_pred_to_gt < tau).float()
    recall_matches = (min_dists_gt_to_pred < tau).float()
    
    precision = precision_matches.sum(dim=1) / N
    recall = recall_matches.sum(dim=1) / M
    
    fscore = (
        2 * (precision * recall) / (precision + recall + 1e-8)
    )
    
    return fscore


def compute_volume_iou(pred, gt, mode="bbox"):
    if mode == "bbox":
        pred_min = pred.min(dim=1).values
        pred_max = pred.max(dim=1).values
        gt_min = gt.min(dim=1).values
        gt_max = gt.max(dim=1).values
        
        intersection_min = torch.max(pred_min, gt_min)
        intersection_max = torch.min(pred_max, gt_max)
        inter_dims = (intersection_max - intersection_min).clamp(min=0)
        inter_vol = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]
        
        pred_dims = (pred_max - pred_min).clamp(min=0)
        pred_vol = pred_dims[:, 0] * pred_dims[:, 1] * pred_dims[:, 2]
        gt_dims = (gt_max - gt_min).clamp(min=0)
        gt_vol = gt_dims[:, 0] * gt_dims[:, 1] * gt_dims[:, 2]
        
        union_vol = pred_vol + gt_vol - inter_vol
        iou = inter_vol / (union_vol + 1e-8)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return iou


def normalize_points(tensor):
    min_vals = tensor.min(dim=1, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0]
    
    ranges = max_vals - min_vals
    ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)
    
    normalized_tensor = 1.9 * (tensor - min_vals) / ranges - 0.95
    
    return normalized_tensor


def sample_points_from_meshes(meshes, num_samples=20000):
    if not isinstance(meshes, list):
        meshes = [meshes]
    
    vertices = []
    for mesh in meshes:
        try:
            vert = trimesh.sample.sample_surface(mesh, num_samples)[0]
        except:
            vert = trimesh.sample.volume_mesh(mesh, num_samples)
        
        vertices.append(torch.from_numpy(vert).float())
    
    return torch.stack(vertices) if len(vertices) > 1 else vertices[0].unsqueeze(0)
