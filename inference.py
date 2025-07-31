import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from dit import DiT
from model import RectifiedFlow
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image


class DiTInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        config = checkpoint.get('config', {})
        self.image_size = config.get('image_size', 64)
        self.image_channels = config.get('image_channels', 3)
        self.sigma_min = config.get('sigma_min', 1e-6)
        
        # Initialize DiT model
        print("Initializing DiT model...")
        self.model = DiT(
            input_size=self.image_size,
            patch_size=2,
            in_channels=self.image_channels,
            dim=384,
            depth=12,
            num_heads=6,
            num_classes=10,
            learn_sigma=False,
            class_dropout_prob=0.1,
        ).to(device)
        
        # Load weights (prefer EMA if available)
        if 'ema' in checkpoint:
            print("Loading EMA weights...")
            self.model.load_state_dict(checkpoint['ema'])
        else:
            print("Loading model weights...")
            self.model.load_state_dict(checkpoint['model'])
        
        self.model.eval()
        
        # Initialize sampler with scheduler parameters from checkpoint
        self.sampler = RectifiedFlow(
            self.model,
            device=device,
            channels=self.image_channels,
            image_size=self.image_size,
            num_classes=10,
            use_logit_normal_cosine=config.get('timestep_sampling', 'logit_normal_cosine') == 'logit_normal_cosine',
            logit_normal_loc=config.get('logit_normal_loc', 0.0),
            logit_normal_scale=config.get('logit_normal_scale', 1.0),
            timestep_min=config.get('timestep_min', 1e-8),
            timestep_max=config.get('timestep_max', 1.0-1e-8),
        )
        
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def sample(self, num_samples=16, class_labels=None, cfg_scale=3.0, 
               num_steps=50, seed=None):
        """
        Sample images from the model.
        
        Args:
            num_samples: Number of samples to generate
            class_labels: List/tensor of class labels (0-9), or None for random
            cfg_scale: Classifier-free guidance scale
            num_steps: Number of sampling steps
            seed: Random seed for reproducibility
        
        Returns:
            Generated images tensor in [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        print(f"Sampling {num_samples} images with CFG scale {cfg_scale}...")
        
        if class_labels is not None:
            # Handle class labels
            if isinstance(class_labels, int):
                class_labels = torch.full((num_samples,), class_labels, device=self.device)
            elif isinstance(class_labels, list):
                class_labels = torch.tensor(class_labels, device=self.device)
            else:
                class_labels = class_labels.to(self.device)
            
            # Ensure correct number of labels
            if len(class_labels) < num_samples:
                class_labels = class_labels.repeat(num_samples // len(class_labels) + 1)[:num_samples]
        
        # Use the sampler's method
        images = self.sampler.sample(
            batch_size=num_samples,
            cfg_scale=cfg_scale,
            sample_steps=num_steps
        )
        
        return images
    
    @torch.no_grad()
    def sample_all_classes(self, samples_per_class=10, cfg_scale=3.0, 
                          num_steps=50, seed=None):
        """Sample images for all classes (0-9)"""
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        print(f"Sampling all classes with {samples_per_class} samples per class...")
        images = self.sampler.sample_each_class(
            n_per_class=samples_per_class,
            cfg_scale=cfg_scale,
            sample_steps=num_steps
        )
        
        return images
    
    @torch.no_grad()
    def sample_grid(self, rows=4, cols=4, cfg_scale=3.0, num_steps=50, 
                   class_labels=None, seed=None):
        """Sample a grid of images"""
        num_samples = rows * cols
        
        images = self.sample(
            num_samples=num_samples,
            class_labels=class_labels,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed
        )
        
        # Create grid
        grid = make_grid(images, nrow=cols, normalize=False)
        return grid
    
    @torch.no_grad()
    def sample_trajectory(self, num_samples=4, class_labels=None, cfg_scale=3.0,
                         num_steps=50, seed=None):
        """Sample images and return the full trajectory for visualization"""
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        print(f"Sampling {num_samples} trajectories with CFG scale {cfg_scale}...")
        
        if class_labels is None:
            class_labels = torch.randint(0, 10, (num_samples,), device=self.device)
        elif isinstance(class_labels, int):
            class_labels = torch.full((num_samples,), class_labels, device=self.device)
        
        # Get samples with trajectory
        final_images, trajectory = self.sampler.sample(
            batch_size=num_samples,
            cfg_scale=cfg_scale,
            sample_steps=num_steps,
            return_all_steps=True
        )
        
        return final_images, trajectory
    
    def save_samples(self, output_dir='outputs', num_samples=64, 
                    cfg_scale=3.0, seed=42):
        """Generate and save sample images"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample all classes grid
        print("Generating class grid...")
        class_grid = self.sample_all_classes(
            samples_per_class=10, 
            cfg_scale=cfg_scale, 
            seed=seed
        )
        grid = make_grid(class_grid, nrow=10, normalize=False)
        save_image(grid, os.path.join(output_dir, f'class_grid_cfg{cfg_scale}.png'))
        
        # Random samples with different CFG scales
        print("Generating random samples...")
        for cfg in [1.0, 2.0, 3.0, 5.0, 7.0]:
            random_grid = self.sample_grid(
                rows=8, 
                cols=8, 
                cfg_scale=cfg, 
                seed=seed
            )
            save_image(random_grid, os.path.join(output_dir, f'random_grid_cfg{cfg}.png'))
        
        # Generate trajectory visualization
        print("Generating trajectory visualization...")
        final_images, trajectory = self.sample_trajectory(
            num_samples=16,
            cfg_scale=cfg_scale,
            num_steps=50,
            seed=seed
        )
        
        # Save a few frames from the trajectory
        trajectory_indices = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, -1]
        trajectory_frames = []
        for idx in trajectory_indices:
            frame = trajectory[idx]
            frame = (frame + 1) / 2  # Convert from [-1, 1] to [0, 1]
            frame = frame.clamp(0, 1)
            trajectory_frames.append(frame)
        
        trajectory_grid = make_grid(torch.cat(trajectory_frames, dim=0), nrow=len(trajectory_indices), normalize=False)
        save_image(trajectory_grid, os.path.join(output_dir, 'trajectory_visualization.png'))
        
        print(f"Samples saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='DiT Direct Image Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                       help='Classifier-free guidance scale')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of sampling steps')
    parser.add_argument('--class_label', type=int, default=None,
                       help='Specific class to sample (0-9), or None for random')
    
    # Action
    parser.add_argument('--action', type=str, default='sample',
                       choices=['sample', 'grid', 'all_classes', 'trajectory', 'all'],
                       help='What to generate')
    
    args = parser.parse_args()
    
    # Initialize model
    model = DiTInference(args.checkpoint, device=args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.action == 'sample':
        # Generate samples
        images = model.sample(
            num_samples=args.num_samples,
            class_labels=args.class_label,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            seed=args.seed
        )
        
        # Save individual images
        for i, img in enumerate(images):
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_np).save(
                os.path.join(args.output_dir, f'sample_{i:04d}.png')
            )
        
        # Save grid
        grid = make_grid(images, nrow=int(np.sqrt(args.num_samples)), normalize=False)
        save_image(grid, os.path.join(args.output_dir, 'samples_grid.png'))
        
    elif args.action == 'grid':
        # Generate grid
        grid = model.sample_grid(
            rows=8, cols=8,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            class_labels=args.class_label,
            seed=args.seed
        )
        save_image(grid, os.path.join(args.output_dir, 'grid.png'))
        
    elif args.action == 'all_classes':
        # Sample all classes
        images = model.sample_all_classes(
            samples_per_class=10,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            seed=args.seed
        )
        grid = make_grid(images, nrow=10, normalize=False)
        save_image(grid, os.path.join(args.output_dir, 'all_classes.png'))
        
    elif args.action == 'trajectory':
        # Generate trajectory visualization
        final_images, trajectory = model.sample_trajectory(
            num_samples=16,
            class_labels=args.class_label,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            seed=args.seed
        )
        
        # Save final images
        grid = make_grid(final_images, nrow=4, normalize=False)
        save_image(grid, os.path.join(args.output_dir, 'trajectory_final.png'))
        
        # Save trajectory frames
        for i, frame_idx in enumerate([0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, -1]):
            frame = trajectory[frame_idx]
            frame = (frame + 1) / 2  # Convert from [-1, 1] to [0, 1]
            frame = frame.clamp(0, 1)
            frame_grid = make_grid(frame, nrow=4, normalize=False)
            save_image(frame_grid, os.path.join(args.output_dir, f'trajectory_frame_{i}.png'))
    
    elif args.action == 'all':
        # Generate all visualizations
        model.save_samples(args.output_dir, num_samples=64, cfg_scale=args.cfg_scale, seed=args.seed)
    
    print(f"\nGenerated images saved to {args.output_dir}/")


if __name__ == "__main__":
    main()