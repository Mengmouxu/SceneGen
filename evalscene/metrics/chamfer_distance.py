import torch

class ChamferDistance(torch.nn.Module):
    def __init__(self, chunk_size=1024):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, xyz1, xyz2):
        B, N, C = xyz1.shape
        _, M, C2 = xyz2.shape
        assert C == 3 and C2 == 3, "Only 3D points supported"
        assert xyz1.device == xyz2.device, "Devices differ"

        device = xyz1.device

        if N == 0 or M == 0:
            return (
                torch.zeros((B, N), dtype=xyz1.dtype, device=device),
                torch.zeros((B, M), dtype=xyz2.dtype, device=device),
            )

        min_dist1 = torch.full((B, N), float('inf'), device=device, dtype=xyz1.dtype)
        chunk = min(self.chunk_size, M)

        j = 0
        while j < M:
            end = min(j + chunk, M)
            diff = dist = None
            try:
                # (B, N, end-j, 3)
                diff = xyz1.unsqueeze(2) - xyz2[:, j:end, :].unsqueeze(1)
                dist = torch.sum(diff * diff, dim=-1)          # (B, N, end-j)
                cur_min, _ = dist.min(dim=2)                   # (B, N)
                min_dist1 = torch.minimum(min_dist1, cur_min)
                j = end
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    chunk //= 2
                    if chunk < 1:
                        raise RuntimeError("Chamfer: cannot reduce chunk size further (xyz2)") from e
                    continue
                raise
            finally:
                if diff is not None:
                    del diff
                if dist is not None:
                    del dist

        min_dist2 = torch.full((B, M), float('inf'), device=device, dtype=xyz2.dtype)
        chunk = min(self.chunk_size, N)
        i = 0
        while i < N:
            end = min(i + chunk, N)
            diff = dist = None
            try:
                diff = xyz2.unsqueeze(2) - xyz1[:, i:end, :].unsqueeze(1)  # (B, M, end-i, 3)
                dist = torch.sum(diff * diff, dim=-1)                      # (B, M, end-i)
                cur_min, _ = dist.min(dim=2)                               # (B, M)
                min_dist2 = torch.minimum(min_dist2, cur_min)
                i = end
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    chunk //= 2
                    if chunk < 1:
                        raise RuntimeError("Chamfer: cannot reduce chunk size further (xyz1)") from e
                    continue
                raise
            finally:
                if diff is not None:
                    del diff
                if dist is not None:
                    del dist

        if torch.isinf(min_dist1).any() or torch.isinf(min_dist2).any():
            print("Warning: Chamfer distance contains inf values, replacing with 0.")
            mask1 = torch.isinf(min_dist1)
            mask2 = torch.isinf(min_dist2)
            if mask1.any():
                min_dist1[mask1] = 0
            if mask2.any():
                min_dist2[mask2] = 0

        return min_dist1, min_dist2