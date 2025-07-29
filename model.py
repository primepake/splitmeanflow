import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from tqdm.auto import tqdm
from dit import DiT


class RectifiedFlowVAE(nn.Module):
    """RectifiedFlow adapted for VAE latent space"""
    def __init__(
        self,
        net: DiT,
        device="cuda",
        channels=4,  # VAE latent channels
        image_size=32,  # Latent size
        num_classes=10,
        logit_normal_sampling_t=True,
    ):
        super().__init__()
        self.net = net
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.logit_normal_sampling_t = logit_normal_sampling_t
    
    @torch.no_grad()
    def sample(self, batch_size, y=None, cfg_scale=5.0, sample_steps=50, return_all_steps=False):
        """Sample in VAE latent space"""
        if self.use_cond and y is None:
            y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        # Start from random noise in latent space
        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        
        images = [z]
        t_span = torch.linspace(0, 1, sample_steps, device=self.device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        
        for t in tqdm(reversed(t_span), desc="Sampling"):
            if self.use_cond:
                v_t = self.net.forward_with_cfg(z, t, y, cfg_scale)
            else:
                v_t = self.net(z, t)
            z = z - v_t / sample_steps
            if return_all_steps:
                images.append(z.clone())
        
        # Return raw latents - no normalization
        # The decode_latents function will handle VAE scaling
        if return_all_steps:
            return z, torch.stack(images)
        return z
    
    @torch.no_grad()
    def sample_each_class(self, n_per_class, cfg_scale=5.0, sample_steps=50, return_all_steps=False):
        """Sample each class in VAE latent space"""
        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        z = torch.randn(self.num_classes * n_per_class, self.channels, self.image_size, self.image_size, device=self.device)
        
        images = [z]
        t_span = torch.linspace(0, 1, sample_steps, device=self.device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        
        for t in tqdm(reversed(t_span), desc="Sampling all classes"):
            v_t = self.net.forward_with_cfg(z, t, c, cfg_scale)
            z = z - v_t / sample_steps
            if return_all_steps:
                images.append(z.clone())
        
        # Return raw latents
        if return_all_steps:
            return z, torch.stack(images)
        return z