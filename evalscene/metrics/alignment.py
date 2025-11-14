import torch
import numpy as np
from probreg import filterreg

def point_alignment(
        src,
        dst,
        max_iterations=50,
        tolerance=1e-4,
        mode='filterreg',
        objective_type='pt2pt',
        update_sigma=True,
        use_cuda=True,
    ):
    if mode == 'filterreg':
        B, N, _ = src.shape
        device = src.device
        
        batch_R = []
        batch_t = []
        transformed_src = torch.empty_like(src)

        for b in range(B):
            
            src_np = src[b].detach().cpu().numpy().astype(np.float64)
            dst_np = dst[b].detach().cpu().numpy().astype(np.float64)

            tf_param, _, _ = filterreg.registration_filterreg(
                src_np, dst_np,
                objective_type=objective_type,
                maxiter=max_iterations,
                tol=tolerance,
                sigma2=None,
                update_sigma2=False,
            )
            
            current_R = torch.tensor(tf_param.rot, dtype=torch.float32, device=device)
            current_t = torch.tensor(tf_param.t, dtype=torch.float32, device=device).view(3, 1)
            
            batch_R.append(current_R)
            batch_t.append(current_t)

            transformed_src[b] = torch.matmul(src[b], current_R.transpose(0, 1)) + current_t.transpose(0, 1)

        R = torch.stack(batch_R, dim=0)
        t = torch.stack(batch_t, dim=0)
        
        return transformed_src, R, t

    if mode == 'cpd':
        try:
            from probreg import cpd as probreg_cpd
        except Exception as e:
            raise ImportError("probreg.cpd is not available; ensure 'probreg' is correctly installed and no local file named 'probreg.py' shadows it.") from e

        cp = np
        to_cpu = lambda x: x

        B, _, _ = src.shape
        device = src.device
        
        batch_R = []
        batch_t = []
        transformed_src = src.clone()

        for b in range(B):
            src_np = src[b].detach().cpu().numpy()
            dst_np = dst[b].detach().cpu().numpy()

            max_points = 5000
            if src_np.shape[0] > max_points:
                idx = np.random.choice(src_np.shape[0], size=max_points, replace=False)
                src_np = src_np[idx]
            if dst_np.shape[0] > max_points:
                idx = np.random.choice(dst_np.shape[0], size=max_points, replace=False)
                dst_np = dst_np[idx]
            
            source_pts = cp.asarray(src_np, dtype=cp.float32)
            target_pts = cp.asarray(dst_np, dtype=cp.float32)
            
            cpd_solver = probreg_cpd.RigidCPD(source=source_pts, use_cuda=False)

            tf_param, _, _ = cpd_solver.registration(
                target=target_pts,
                maxiter=max_iterations,
                tol=tolerance
            )
            
            rot_cpu = to_cpu(tf_param.rot)
            t_cpu = to_cpu(tf_param.t)
            scale_cpu = to_cpu(tf_param.scale)

            current_R = torch.tensor(rot_cpu, dtype=torch.float32, device=device)
            current_t = torch.tensor(t_cpu, dtype=torch.float32, device=device).view(3, 1)
            current_s = torch.tensor(scale_cpu, dtype=torch.float32, device=device).view(1, 1)
            
            batch_R.append(current_R)
            batch_t.append(current_t)

            transformed_src[b] = torch.matmul(src[b], current_R.transpose(0, 1)) * current_s + current_t.transpose(0, 1)

        R = torch.stack(batch_R, dim=0)
        t = torch.stack(batch_t, dim=0)
        
        return transformed_src, R, t