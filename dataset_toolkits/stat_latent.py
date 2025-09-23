import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16',
                        help='Latent model to use')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='Number of samples to use for calculating stats')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.filter_low_aesthetic_score is not None:
        metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
    metadata = metadata[metadata[f'latent_{opt.model}'] == True]
    sha256s = metadata['sha256'].values
    sha256s = np.random.choice(sha256s, min(opt.num_samples, len(sha256s)), replace=False)

    # stats
    means = []
    mean2s = []
    with ThreadPoolExecutor(max_workers=16) as executor, \
        tqdm(total=len(sha256s), desc="Extracting features") as pbar:
        def worker(sha256):
            try:
                feats = np.load(os.path.join(opt.output_dir, 'latents', opt.model, f'{sha256}.npz'))
                feats = feats['feats']
                means.append(feats.mean(axis=0))
                mean2s.append((feats ** 2).mean(axis=0))
                pbar.update()
            except Exception as e:
                print(f"Error extracting features for {sha256}: {e}")
                pbar.update()

        executor.map(worker, sha256s)
        executor.shutdown(wait=True)

    mean = np.array(means).mean(axis=0)
    mean2 = np.array(mean2s).mean(axis=0)
    std = np.sqrt(mean2 - mean ** 2)

    print('mean:', mean)
    print('std:', std)

    with open(os.path.join(opt.output_dir, 'latents', opt.model, 'stats.json'), 'w') as f:
        json.dump({
            'mean': mean.tolist(),
            'std': std.tolist(),
        }, f, indent=4)
        