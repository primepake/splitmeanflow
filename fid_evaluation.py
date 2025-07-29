import math
import os

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm


def num_to_groups(n, divisor):
    groups = n // divisor
    remainder = n % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class FIDEvaluation:
    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        num_classes=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=10000,
        inception_block_idx=2048,
    ):
        self.batch_size = batch_size
        self.dl = dl
        self.sampler = sampler
        self.print_fn = print
        self.channels = channels
        self.num_classes = num_classes
        self.stats_dir = stats_dir
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir, exist_ok=True)
        self.device = device
        self.n_samples = num_fid_samples
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False
    
    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples = next(self.dl)
                except StopIteration:
                    break
                if len(real_samples) != 1:
                    real_samples = real_samples[0]
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True
    
    @torch.inference_mode()
    def fid_score(self, cfg_scale=5.0, sample_steps=25):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )
        for batch in tqdm(batches):
            fake_samples = self.sampler.sample(batch_size=batch, cfg_scale=cfg_scale, sample_steps=sample_steps)
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)