import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from diffusers.models import AutoencoderKL
from dit import DiT
from model import RectifiedFlowVAE as RectifiedFlow
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image


class DiTVAEInference:
    def __init__(self, checkpoint_path, device='cuda', vae_model='ema'):
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        config = checkpoint.get('config', {})
        self.vae_model = config.get('vae_model', vae_model)
        self.image_size = config.get('image_size', 256)
        self.latent_size = config.get('latent_size', 32)
        self.latent_channels = config.get('latent_channels', 4)
        
        # Load VAE
        print(f"Loading VAE model (sd-vae-ft-{self.vae_model})...")
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.vae_model}").to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        # Initialize DiT model
        print("Initializing DiT model...")
        self.model = DiT(
            input_size=self.latent_size,
            patch_size=2,
            in_channels=self.latent_channels,
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
        
        # Initialize sampler
        self.sampler = RectifiedFlow(
            self.model,
            device=device,
            channels=self.latent_channels,
            image_size=self.latent_size,
            num_classes=10,
        )
        
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to images using VAE"""
        # Denormalize latents
        latents = latents / 0.18215
        images = self.vae.decode(latents).sample
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        return images.clamp(0, 1)
    
    @torch.no_grad()
    def sample(self, num_samples=16, class_labels=None, cfg_scale=3.0, 
               num_steps=50, seed=None, return_latents=False):
        """
        Sample images from the model.
        
        Args:
            num_samples: Number of samples to generate
            class_labels: List/tensor of class labels (0-9), or None for random
            cfg_scale: Classifier-free guidance scale
            num_steps: Number of sampling steps
            seed: Random seed for reproducibility
            return_latents: If True, return latents instead of decoded images
        
        Returns:
            Generated images tensor (or latents if return_latents=True)
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Sample latents using RectifiedFlow's sample method
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
        latents = self.sampler.sample(
            batch_size=num_samples,
            y=class_labels,  # Pass None for random
            cfg_scale=cfg_scale,
            sample_steps=num_steps
        )
        
        if return_latents:
            return latents
        
        # Decode to images
        images = self.decode_latents(latents)
        return images
    
    @torch.no_grad()
    def sample_all_classes(self, samples_per_class=10, cfg_scale=3.0, 
                          num_steps=50, seed=None):
        """Sample images for all classes (0-9)"""
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Use the sampler's built-in sample_each_class method
        print(f"Sampling all classes with {samples_per_class} samples per class...")
        latents = self.sampler.sample_each_class(
            n_per_class=samples_per_class,
            cfg_scale=cfg_scale,
            sample_steps=num_steps
        )
        
        # Decode to images
        images = self.decode_latents(latents)
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
    def interpolate_classes(self, class1, class2, num_steps=10, 
                           cfg_scale=3.0, sampling_steps=50, seed=None):
        """Interpolate between two classes in latent space"""
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # For DiT with discrete classes, we'll create a smooth transition
        # by gradually changing the class conditioning
        interpolated_images = []
        
        # Sample with fixed initial noise
        z_init = torch.randn(1, self.latent_channels, self.latent_size, 
                            self.latent_size, device=self.device)
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            # Use the same initial noise for consistency
            z = z_init.clone()
            
            # Determine which class to use (can be extended to class mixing)
            if alpha < 0.5:
                current_class = torch.tensor([class1], device=self.device)
            else:
                current_class = torch.tensor([class2], device=self.device)
            
            # Sample with fixed noise
            t_span = torch.linspace(0, 1, sampling_steps, device=self.device)
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
            
            for t in reversed(t_span):
                v_t = self.model.forward_with_cfg(z, t, current_class, cfg_scale)
                z = z - v_t / sampling_steps
            
            # Decode the latent
            image = self.decode_latents(z)
            interpolated_images.append(image)
        
        return torch.cat(interpolated_images, dim=0)
    
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
        
        # Random samples
        print("Generating random samples...")
        for cfg in [1.0, 2.0, 3.0, 5.0]:
            random_grid = self.sample_grid(
                rows=8, 
                cols=8, 
                cfg_scale=cfg, 
                seed=seed
            )
            save_image(random_grid, os.path.join(output_dir, f'random_grid_cfg{cfg}.png'))
        
        print(f"Samples saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='DiT VAE Inference')
    parser.add_argument('--checkpoint', type=str, required=False, default='/mnt/nvme/checkpoint/dit_vae/step_0.pth',
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
                       choices=['sample', 'grid', 'all_classes', 'interpolate'],
                       help='What to generate')
    
    args = parser.parse_args()
    
    # Initialize model
    model = DiTVAEInference(args.checkpoint, device=args.device)
    
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
        
    elif args.action == 'interpolate':
        # Interpolate between classes
        if args.class_label is None:
            class1, class2 = 0, 9  # Default: interpolate between first and last class
        else:
            class1 = args.class_label
            class2 = (args.class_label + 1) % 10
        
        images = model.interpolate_classes(
            class1, class2,
            num_steps=10,
            cfg_scale=args.cfg_scale,
            sampling_steps=args.num_steps,
            seed=args.seed
        )
        grid = make_grid(images, nrow=10, normalize=False)
        save_image(grid, os.path.join(args.output_dir, f'interpolate_{class1}_to_{class2}.png'))
    
    print(f"Generated images saved to {args.output_dir}/")


if __name__ == "__main__":
    main()