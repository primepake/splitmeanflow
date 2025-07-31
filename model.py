import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
from functools import partial
from einops import rearrange
import math

from tqdm.auto import tqdm

from dit import DiT


class LogitNormalCosineScheduler:
    """
    Combined Logit-Normal timestep sampling + Cosine interpolation scheduling.
    This is the optimal approach used in Stable Diffusion 3.
    """
    
    def __init__(self, loc: float = 0.0, scale: float = 1.0, min_t: float = 1e-4, max_t: float = 1.0 - 1e-4):
        """
        Args:
            loc: Location parameter (mu) for logit-normal - 0.0 for symmetric around t=0.5
            scale: Scale parameter (sigma) for logit-normal - 1.0 is SD3 default
            min_t: Minimum timestep to avoid singularities
            max_t: Maximum timestep to avoid singularities
        """
        self.loc = loc
        self.scale = scale
        self.min_t = min_t
        self.max_t = max_t
        # Create LogitNormal using TransformedDistribution
        base_normal = Normal(loc, scale)
        self.logit_normal = TransformedDistribution(base_normal, SigmoidTransform())
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from logit-normal distribution."""
        # Step 1: Sample from logit-normal distribution
        t = self.logit_normal.sample((batch_size,)).to(device)
        
        # Step 2: Clamp to avoid singularities at 0 and 1
        t = torch.clamp(t, self.min_t, self.max_t)
        
        return t
    
    def get_cosine_schedule_params(self, t: torch.Tensor, sigma_min: float = 1e-6) -> tuple:
        """
        Convert logit-normal sampled timesteps to cosine-scheduled interpolation parameters.
        
        Args:
            t: Logit-normal sampled timesteps [batch_size]
            sigma_min: Minimum noise level
            
        Returns:
            alpha_t, sigma_t: Cosine-scheduled interpolation parameters
        """
        # Apply cosine scheduling transformation
        t_cos = 0.5 * (1 - torch.cos(math.pi * t))
        
        # Cosine interpolation parameters
        alpha_t = t_cos
        # sigma_t = 1 - t_cos + sigma_min * t_cos
        sigma_t = 1 - t_cos * (1 - sigma_min)
        
        return alpha_t, sigma_t

    def get_velocity_target(self, x1: torch.Tensor, z: torch.Tensor, sigma_min: float = 1e-6) -> torch.Tensor:
        """
        Compute velocity target for cosine-scheduled flow matching.
        
        Args:
            x1: Clean data
            z: Noise  
            t: Logit-normal sampled timesteps
            sigma_min: Minimum noise level
            
        Returns:
            u: Velocity target
        """
        # For cosine scheduling, the velocity target is:
        u = x1 - (1 - sigma_min) * z
        return u

    def create_cosine_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Create cosine-scheduled timestep sequence for inference."""
        t_span = torch.linspace(0, 1, num_steps + 1, device=device)
        # Apply cosine transformation for smoother scheduling
        t_span = 0.5 * (1 - torch.cos(math.pi * t_span))
        return t_span


def normalize_to_neg1_1(x):
    return x * 2 - 1

def unnormalize_to_0_1(x):
    return (x + 1) * 0.5


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        net: DiT,
        device="cuda",
        channels=3,
        image_size=32,
        num_classes=10,
        logit_normal_sampling_t=True,  # Kept for backward compatibility
        use_logit_normal_cosine=True,
        # Scheduler parameters
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
        timestep_min=1e-4,
        timestep_max=1.0-1e-4,
    ):
        super().__init__()
        self.net = net
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.logit_normal_sampling_t = logit_normal_sampling_t
        self.use_logit_normal_cosine = use_logit_normal_cosine
        
        # FIXED: Initialize scheduler properly
        if self.use_logit_normal_cosine:
            self.scheduler = LogitNormalCosineScheduler(
                loc=logit_normal_loc,
                scale=logit_normal_scale,
                min_t=timestep_min,
                max_t=timestep_max
            )
        else:
            self.scheduler = None

    def forward(self, x, c=None):
        """I used forward directly instead of via sampler"""
        pass

    def get_timestep_schedule(self, sample_steps: int):
        """Get timestep schedule based on configuration."""
        if self.use_logit_normal_cosine and self.scheduler is not None:
            return self.scheduler.create_cosine_schedule(sample_steps, self.device)
        else:
            # Fallback to linear schedule
            return torch.linspace(0, 1, sample_steps + 1, device=self.device)
    
    @torch.no_grad()
    def sample(self, batch_size=None, class_labels=None, cfg_scale=5.0, sample_steps=10, return_all_steps=False):
        """
        Sample images using configured scheduling.
        
        Args:
            batch_size: Number of samples to generate (required if class_labels is None)
            class_labels: Tensor of class labels to condition on (optional)
            cfg_scale: Classifier-free guidance scale
            sample_steps: Number of sampling steps
            return_all_steps: Whether to return all intermediate steps
        
        Returns:
            Generated samples in [0, 1] range
        """
        # Determine batch size and conditioning
        if class_labels is not None:
            # Use provided class labels
            batch_size = class_labels.shape[0]
            c = class_labels.to(self.device)
        elif self.use_cond and batch_size is not None:
            # Generate random class labels
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        elif batch_size is not None:
            # No conditioning
            c = None
        else:
            raise ValueError("Either batch_size or class_labels must be provided")
        
        print('class labels: ', c)
        
        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        
        images = []
        if return_all_steps:
            images.append(z.clone())

        t_span = self.get_timestep_schedule(sample_steps)

        t = t_span[0]
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            if self.use_cond and c is not None:
                print(f"sample using cfg_scale: {cfg_scale}")
                v_t = self.net.forward_with_cfg(z, t, c, cfg_scale)
            else:
                v_t = self.net(z, t)

            z = z + dt * v_t
            t = t + dt
            
            # Store intermediate result
            if return_all_steps:
                images.append(z.clone())
            
            # Update dt for next step
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        
        # z_final = unnormalize_to_0_1(z.clip(-1, 1))
        z_final = z.clip(-1, 1)
        # z_final = z

        if return_all_steps:
            # Return both final image and full trajectory
            return z_final, torch.stack(images)
        return z_final
        
    
    @torch.no_grad()
    def sample_each_class(self, n_per_class, cfg_scale=5.0, sample_steps=10, return_all_steps=False):
        """Sample n_per_class images for each class."""
        if not self.use_cond:
            raise ValueError("Cannot sample each class when num_classes is None")
        
        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        print('class: ', c)
        z = torch.randn(self.num_classes * n_per_class, self.channels, self.image_size, self.image_size, device=self.device)
        
        # FIXED: Consistent trajectory tracking
        images = [z.clone()] if return_all_steps else []
        t_span = self.get_timestep_schedule(sample_steps)
        
        t = t_span[0]
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            if self.use_cond:
                print(f"Using cfg_scale: {cfg_scale}")
                v_t = self.net.forward_with_cfg(z, t, c, cfg_scale)
            else:
                v_t = self.net(z, t)

            z = z + dt * v_t
            t = t + dt
            
            # Store intermediate result
            if return_all_steps:
                images.append(z.clone())
            
            # Update dt for next step
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        
        # z_final = unnormalize_to_0_1(z.clip(-1, 1))
        z_final = z.clip(-1, 1)
        # z_final = z
    
        if return_all_steps:
            # Return both final image and full trajectory
            return z_final, torch.stack(images)  # Keep trajectory in [-1, 1] for GIF creation
        return z_final

    @classmethod
    def from_checkpoint(cls, checkpoint_path, net, device="cuda"):
        """
        Create RectifiedFlow sampler from training checkpoint.
        Automatically loads the correct scheduler parameters.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Extract scheduler parameters from checkpoint
        use_logit_normal_cosine = config.get('timestep_sampling') == 'logit_normal_cosine'
        
        sampler = cls(
            net=net,
            device=device,
            channels=config.get('image_channels', 3),
            image_size=config.get('image_size', 32),
            num_classes=config.get('num_classes', 10),
            use_logit_normal_cosine=use_logit_normal_cosine,
            logit_normal_loc=config.get('logit_normal_loc', 0.0),
            logit_normal_scale=config.get('logit_normal_scale', 1.0),
            timestep_min=config.get('timestep_min', 1e-4),
            timestep_max=config.get('timestep_max', 1.0-1e-4),
        )
        
        return sampler


# Example usage
if __name__ == "__main__":
    # Test the fixed implementation
    from dit import DiT
    
    model = DiT(
        input_size=64,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10
    )
    
    # Create sampler with logit-normal + cosine scheduling
    sampler = RectifiedFlow(
        net=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        channels=3,
        image_size=64,
        num_classes=10,
        use_logit_normal_cosine=True,
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
    )
    
    # Test sampling
    with torch.no_grad():
        samples = sampler.sample(batch_size=4, sample_steps=20)
        print(f"Generated samples shape: {samples.shape}")
        
        class_samples = sampler.sample_each_class(n_per_class=2, sample_steps=20)
        print(f"Class samples shape: {class_samples.shape}")