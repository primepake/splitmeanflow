import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from tqdm.auto import tqdm

from dit import DiT


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

    def forward(self, x, c=None):
        """ I used forward directly instad of via sampler """
        pass
    
    @torch.no_grad()
    def sample(self, batch_size, cfg_scale=5.0, sample_steps=10, return_all_steps=False):
        if self.use_cond:
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        
        images = [z]
        t_span = torch.linspace(0, 1, sample_steps+1, device=self.device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            if self.use_cond:
                v_t = self.net.forward_with_cfg(z, t, c, cfg_scale)
            else:
                v_t = self.net(z, t)

            z = z + dt * v_t
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

            images.append(z)
        
        z = unnormalize_to_0_1(z.clip(-1, 1))
        
        if return_all_steps:
            return z, torch.stack(images)
        return z
        
    
    @torch.no_grad()
    def sample_each_class(self, n_per_class, cfg_scale=5.0, sample_steps=10, return_all_steps=False):
        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        z = torch.randn(self.num_classes * n_per_class, self.channels, self.image_size, self.image_size, device=self.device)
        
        images = []
        t_span = torch.linspace(0, 1, sample_steps+1, device=self.device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            if self.use_cond:
                v_t = self.net.forward_with_cfg(z, t, c, cfg_scale)
            else:
                v_t = self.net(z, t)

            z = z + dt * v_t
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

            images.append(z)
        
        z = unnormalize_to_0_1(z.clip(-1, 1))
        
        if return_all_steps:
            return z, torch.stack(images)
        return z