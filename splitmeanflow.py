import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from tqdm.auto import tqdm
from dit import DiT
import copy


def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5


class SplitMeanFlowDiT(DiT):
    """
    Modified DiT that accepts interval inputs (r, t) instead of just t.
    Inherits from the original DiT and adds interval embedding.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional embedding for interval information
        dim = self.t_embedder.mlp[-1].out_features
        self.interval_embedder = nn.Sequential(
            nn.Linear(2, dim),  # [r, t] -> dim
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x, r, t, y):
        """
        Forward pass with interval inputs.
        x: (N, C, H, W) tensor of spatial inputs
        r: (N,) tensor of interval start times
        t: (N,) tensor of interval end times
        y: (N,) tensor of class labels
        """
        H, W = x.shape[-2:]
        x_emb = self.x_embedder(x) + self.pos_embed
        
        # Pack with register tokens
        r_tokens = repeat(self.register_tokens, 'n d -> b n d', b=x_emb.shape[0])
        x_emb, ps = pack([x_emb, r_tokens], 'b * d ')
        
        # Time embeddings
        t_emb = self.t_embedder(t)
        
        # Interval embedding: encode [r, t] information
        interval = torch.stack([r, t], dim=-1)  # (N, 2)
        interval_emb = self.interval_embedder(interval)  # (N, D)
        
        # Combine embeddings
        c = t_emb + interval_emb
        if self.use_cond:
            y_emb = self.y_embedder(y, self.training)
            c = c + y_emb
        
        # Process through transformer blocks
        for block in self.blocks:
            x_emb = block(x_emb, c)
        
        # Unpack and final layer
        x_emb, _ = unpack(x_emb, ps, 'b * d')
        x_out = self.final_layer(x_emb, c)
        x_out = self.unpatchify(x_out)
        
        return x_out
    
    def forward_with_cfg(self, x, r, t, y, cfg_scale):
        """CFG forward for interval inputs"""
        x = x.repeat(2, 1, 1, 1)
        r = r.repeat(2)
        t = t.repeat(2)
        y = y.repeat(2)
        y[len(x) // 2:] = self.y_embedder.num_classes
        
        model_out = self.forward(x, r, t, y)
        cond_out, uncond_out = model_out.split(len(x) // 2)
        out = uncond_out + (cond_out - uncond_out) * cfg_scale
        return out


class SplitMeanFlow(nn.Module):
    def __init__(
        self,
        student_net: SplitMeanFlowDiT,
        teacher_model: nn.Module,  # This is RectifiedFlow which contains teacher.net
        device="cuda",
        channels=3,
        image_size=32,
        num_classes=10,
        flow_ratio=0.5,  # Fraction of boundary condition training
        cfg_scale=3.0,   # Fixed CFG scale for teacher
        time_sampling="uniform",  # or "lognormal"
    ):
        super().__init__()
        self.student = student_net
        self.teacher = teacher_model  # This is RectifiedFlow
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.flow_ratio = flow_ratio
        self.cfg_scale = cfg_scale
        self.time_sampling = time_sampling
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def sample_times(self, batch_size):
        """Sample r < t, with optional importance sampling"""
        if self.time_sampling == "lognormal":
            # Sample from log-normal distribution
            times = torch.randn(batch_size, 2, device=self.device).sigmoid()
        else:
            # Uniform sampling
            times = torch.rand(batch_size, 2, device=self.device)
        
        r, t = times.min(dim=1)[0], times.max(dim=1)[0]
        return r, t
    
    def forward(self, x, c=None):
        """
        Training forward pass implementing Algorithm 1 from SplitMeanFlow paper.
        """
        batch_size = x.shape[0]
        x = normalize_to_neg1_1(x)
        
        # Decide whether to use boundary condition or interval splitting
        if torch.rand(1).item() < self.flow_ratio:
            # BOUNDARY CONDITION: u(z_t, t, t) = v(z_t, t)
            t = torch.rand(batch_size, device=self.device)
            t_ = rearrange(t, "b -> b 1 1 1")
            
            # Create interpolated point
            z = torch.randn_like(x)
            z_t = (1 - t_) * x + t_ * z
            
            # Get teacher's velocity (with CFG if needed)
            with torch.no_grad():
                if self.cfg_scale > 0 and self.use_cond:
                    v_teacher = self.teacher.net.forward_with_cfg(z_t, t, c, self.cfg_scale)
                else:
                    v_teacher = self.teacher.net(z_t, t, c)
            
            # Student predicts average velocity u(z_t, t, t) = v(z_t, t)
            u_pred = self.student(z_t, t, t, c)  # r = t
            
            loss = F.mse_loss(u_pred, v_teacher)
            
        else:
            # INTERVAL SPLITTING CONSISTENCY
            # Sample r < t
            r, t = self.sample_times(batch_size)
            
            # Sample λ and compute s
            lam = torch.rand(batch_size, device=self.device)
            s = (1 - lam) * t + lam * r
            
            # Create z_t
            t_ = rearrange(t, "b -> b 1 1 1")
            z = torch.randn_like(x)
            z_t = (1 - t_) * x + t_ * z
            
            # Forward pass 1: u(z_t, s, t)
            u2 = self.student(z_t, s, t, c)
            
            # Compute z_s using the predicted velocity
            t_s_diff = rearrange(t - s, "b -> b 1 1 1")
            z_s = z_t - t_s_diff * u2
            
            # Forward pass 2: u(z_s, r, s)
            u1 = self.student(z_s, r, s, c)
            
            # Construct target using interval splitting consistency
            lam_ = rearrange(lam, "b -> b 1 1 1")
            target = (1 - lam_) * u1 + lam_ * u2
            
            # Forward pass 3: predict u(z_t, r, t)
            u_pred = self.student(z_t, r, t, c)
            
            # Loss with stop gradient on target
            loss = F.mse_loss(u_pred, target.detach())
        
        return loss
    
    @torch.no_grad()
    def sample_onestep(self, batch_size, c=None, cfg_scale=None):
        """One-step generation: z_0 = z_1 - u(z_1, 0, 1)"""
        # Sample initial noise
        z_1 = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.device)
        
        # Create time tensors
        r = torch.zeros(batch_size, device=self.device)
        t = torch.ones(batch_size, device=self.device)
        
        # Generate class labels if needed
        if self.use_cond and c is None:
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        # Predict average velocity over [0, 1]
        if cfg_scale is not None and cfg_scale > 0:
            u = self.student.forward_with_cfg(z_1, r, t, c, cfg_scale)
        else:
            u = self.student(z_1, r, t, c)
        
        # One-step update
        z_0 = z_1 - u
        
        # Denormalize to [0, 1]
        z_0 = unnormalize_to_0_1(z_0.clamp(-1, 1))
        return z_0
    
    @torch.no_grad()
    def sample_twostep(self, batch_size, c=None, cfg_scale=None):
        """Two-step generation for potentially better quality"""
        # Sample initial noise
        z_1 = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.device)
        
        # Generate class labels if needed
        if self.use_cond and c is None:
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        # Step 1: z_1 -> z_0.5
        r1 = torch.full((batch_size,), 0.5, device=self.device)
        t1 = torch.ones(batch_size, device=self.device)
        
        if cfg_scale is not None and cfg_scale > 0:
            u1 = self.student.forward_with_cfg(z_1, r1, t1, c, cfg_scale)
        else:
            u1 = self.student(z_1, r1, t1, c)
        
        z_05 = z_1 - 0.5 * u1
        
        # Step 2: z_0.5 -> z_0
        r2 = torch.zeros(batch_size, device=self.device)
        t2 = torch.full((batch_size,), 0.5, device=self.device)
        
        if cfg_scale is not None and cfg_scale > 0:
            u2 = self.student.forward_with_cfg(z_05, r2, t2, c, cfg_scale)
        else:
            u2 = self.student(z_05, r2, t2, c)
        
        z_0 = z_05 - 0.5 * u2
        
        # Denormalize to [0, 1]
        z_0 = unnormalize_to_0_1(z_0.clamp(-1, 1))
        return z_0
    
    @torch.no_grad()
    def sample_each_class(self, n_per_class, num_steps=1, cfg_scale=None):
        """Sample from each class"""
        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        batch_size = self.num_classes * n_per_class
        
        if num_steps == 1:
            return self.sample_onestep(batch_size, c, cfg_scale)
        elif num_steps == 2:
            return self.sample_twostep(batch_size, c, cfg_scale)
        else:
            raise ValueError(f"Only 1 or 2 step sampling supported, got {num_steps}")


# Helper function to create student from teacher
def create_student_from_teacher(teacher_dit, teacher_checkpoint_path=None):
    """
    Create a SplitMeanFlow student model from a trained teacher DiT.
    """
    # Create student with same architecture
    student = SplitMeanFlowDiT(
        input_size=teacher_dit.x_embedder.img_size[0],
        patch_size=teacher_dit.patch_size,
        in_channels=teacher_dit.in_channels,
        dim=teacher_dit.t_embedder.mlp[-1].out_features,
        depth=len(teacher_dit.blocks),
        num_heads=teacher_dit.num_heads,
        num_register_tokens=teacher_dit.register_tokens.shape[0],
        class_dropout_prob=0.0,  # No dropout for student!
        num_classes=teacher_dit.num_classes,
        learn_sigma=teacher_dit.learn_sigma,
    )
    
    # Initialize student from teacher weights
    # Copy all matching parameters
    teacher_state = teacher_dit.state_dict()
    student_state = student.state_dict()
    
    for name, param in teacher_state.items():
        if name in student_state and param.shape == student_state[name].shape:
            student_state[name].copy_(param)
    
    student.load_state_dict(student_state)
    
    return student