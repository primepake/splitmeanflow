import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from tqdm.auto import tqdm
from dit import SMDiT
import copy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import pack, unpack
from model import LogitNormalCosineScheduler  # Import from teacher's model
from dit import SMDiT


def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5



class SplitMeanFlow(nn.Module):
    def __init__(
        self,
        student_net: SMDiT,
        teacher_model: nn.Module,
        device="cuda",
        channels=3,
        image_size=16,
        num_classes=10,
        flow_ratio=0.5,
        cfg_scale=3.0,
        time_sampling="logit_normal_cosine",
        sigma_min=1e-06,
        use_immiscible=True,
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
        timestep_min=1e-8,
        timestep_max=1.0-1e-8,
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
        self.sigma_min = sigma_min
        self.use_immiscible = use_immiscible
        
        # Initialize the same scheduler as teacher if using logit-normal + cosine
        if self.time_sampling == "logit_normal_cosine":
            self.scheduler = LogitNormalCosineScheduler(
                loc=logit_normal_loc,
                scale=logit_normal_scale,
                min_t=timestep_min,
                max_t=timestep_max
            )
        else:
            self.scheduler = None
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def sample_immiscible_noise(self, x, k=4):
        """Sample noise using immiscible diffusion (farthest from data)"""
        b, c, h, w = x.shape
        z_candidates = torch.randn(b, k, c, h, w, device=x.device, dtype=x.dtype)
        
        x_flat = x.flatten(start_dim=1)  # [b, c*h*w]
        z_candidates_flat = z_candidates.flatten(start_dim=2)  # [b, k, c*h*w]
        
        # Compute distances between each data point and its k noise candidates
        distances = torch.norm(x_flat.unsqueeze(1) - z_candidates_flat, dim=2)  # [b, k]
        
        # Find the farthest noise sample for each data point
        min_distances, min_indices = torch.min(distances, dim=1)  # [b]
        
        batch_indices = torch.arange(b, device=x.device)
        z = z_candidates[batch_indices, min_indices]  # [b, c, h, w]
        return z
    
    def sample_times(self, batch_size):
        """Sample r < t using the appropriate schedule"""
        if self.time_sampling == "logit_normal_cosine" and self.scheduler is not None:
            # Use logit-normal sampling like teacher
            r = self.scheduler.sample_timesteps(batch_size, self.device)
            t = self.scheduler.sample_timesteps(batch_size, self.device)
            
            # Ensure r < t
            r_min = torch.minimum(r, t)
            t_max = torch.maximum(r, t)
            return r_min, t_max
        elif self.time_sampling == "cosine":
            # Simple cosine schedule
            times = torch.rand(batch_size, 2, device=self.device)
            times = 1 - torch.cos(times * 0.5 * torch.pi)
            r, t = times.min(dim=1)[0], times.max(dim=1)[0]
        else:
            # Uniform sampling
            times = torch.rand(batch_size, 2, device=self.device)
            r, t = times.min(dim=1)[0], times.max(dim=1)[0]
        
        # Ensure r < t (not just r <= t)
        mask = (r == t)
        t[mask] = torch.minimum(t[mask] + 0.01, torch.ones_like(t[mask]))
        return r, t
    
    def forward(self, x, c=None):
        """
        Training forward pass implementing Algorithm 1 from SplitMeanFlow paper.
        x is assumed to be in [-1, 1] already from dataloader
        """
        batch_size = x.shape[0]
        rand_val = torch.rand(1).item()
        
        if rand_val < self.flow_ratio:
            # BOUNDARY CONDITION: u(z_t, t, t) = v(z_t, t)
            if self.time_sampling == "logit_normal_cosine" and self.scheduler is not None:
                # Use the same sampling as teacher
                t = self.scheduler.sample_timesteps(batch_size, self.device)
                # Get cosine-scheduled interpolation parameters
                alpha_t, sigma_t = self.scheduler.get_cosine_schedule_params(t, self.sigma_min)
                alpha_t = alpha_t.view(batch_size, 1, 1, 1)
                sigma_t = sigma_t.view(batch_size, 1, 1, 1)
            else:
                t = torch.rand(batch_size, device=self.device)
                t_ = rearrange(t, "b -> b 1 1 1")
                alpha_t = t_
                sigma_t = 1 - (1 - self.sigma_min) * t_
            
            # Sample noise
            if self.use_immiscible:
                z = self.sample_immiscible_noise(x)
            else:
                z = torch.randn_like(x)
            
            # Interpolate using the same formula as teacher
            z_t = sigma_t * z + alpha_t * x
            
            # Get teacher's velocity (with CFG if needed)
            with torch.no_grad():
                if self.cfg_scale > 0 and self.use_cond:
                    print('t before shape: ', t.shape)
                    v_teacher = self.teacher.net.forward_with_cfg(z_t, t, c, self.cfg_scale, is_train_student=True)
                else:
                    v_teacher = self.teacher.net(z_t, t, c)
            
            # Student predicts average velocity u(z_t, t, t) = v(z_t, t)
            u_pred = self.student(z_t, t, t, c)  # r = t
            
            loss = F.mse_loss(u_pred, v_teacher)
            return loss, "boundary"

        elif rand_val < self.flow_ratio + 0.1:

            if torch.rand(1).item() < 0.33:
                # Full interval [0, 1]
                r = torch.zeros(batch_size, device=self.device)
                t = torch.ones(batch_size, device=self.device)
            elif torch.rand(1).item() < 0.66:
                # First half [0, 0.5]
                r = torch.zeros(batch_size, device=self.device)
                t = torch.full((batch_size,), 0.5, device=self.device)
            else:
                # Second half [0.5, 1]
                r = torch.full((batch_size,), 0.5, device=self.device)
                t = torch.ones(batch_size, device=self.device)
            
            lam = torch.rand(batch_size, device=self.device)
            s = (1 - lam) * t + lam * r

            # Get interpolation parameters for z_t
            if self.time_sampling == "logit_normal_cosine" and self.scheduler is not None:
                alpha_t, sigma_t = self.scheduler.get_cosine_schedule_params(t, self.sigma_min)
                alpha_t = alpha_t.view(batch_size, 1, 1, 1)
                sigma_t = sigma_t.view(batch_size, 1, 1, 1)
            else:
                t_ = rearrange(t, "b -> b 1 1 1")
                alpha_t = t_
                sigma_t = 1 - (1 - self.sigma_min) * t_
            
            # Sample noise
            if self.use_immiscible:
                z = self.sample_immiscible_noise(x)
            else:
                z = torch.randn_like(x)
            
            # Create z_t using the same interpolation as teacher
            z_t = sigma_t * z + alpha_t * x
            
            # Forward pass 1: u(z_t, s, t)
            u2 = self.student(z_t, s, t, c)
            
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

            return loss, "consistency"
        else:
            # INTERVAL SPLITTING CONSISTENCY
            # Sample r < t
            r, t = self.sample_times(batch_size)
            
            # Sample λ and compute s
            lam = torch.rand(batch_size, device=self.device)
            s = (1 - lam) * t + lam * r
            
            # Get interpolation parameters for z_t
            if self.time_sampling == "logit_normal_cosine" and self.scheduler is not None:
                alpha_t, sigma_t = self.scheduler.get_cosine_schedule_params(t, self.sigma_min)
                alpha_t = alpha_t.view(batch_size, 1, 1, 1)
                sigma_t = sigma_t.view(batch_size, 1, 1, 1)
            else:
                t_ = rearrange(t, "b -> b 1 1 1")
                alpha_t = t_
                sigma_t = 1 - (1 - self.sigma_min) * t_
            
            # Sample noise
            if self.use_immiscible:
                z = self.sample_immiscible_noise(x)
            else:
                z = torch.randn_like(x)
            
            # Create z_t using the same interpolation as teacher
            z_t = sigma_t * z + alpha_t * x
            
            # Forward pass 1: u(z_t, s, t)
            u2 = self.student(z_t, s, t, c)
            
            # Compute z_s using the predicted velocity
            t_s_diff = t_s_diff = rearrange(t - s, "b -> b 1 1 1")
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

            return loss, "consistency"
    
    @torch.no_grad()
    def sample(self, batch_size=None, class_labels=None, num_steps=1, cfg_scale=None):
        """General sampling method"""
        if class_labels is not None:
            batch_size = class_labels.shape[0]
            c = class_labels
        elif self.use_cond and batch_size is not None:
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        else:
            c = None
        
        if num_steps == 1:
            return self.sample_onestep(batch_size, c, cfg_scale)
        elif num_steps == 2:
            return self.sample_twostep(batch_size, c, cfg_scale)
        else:
            return self.sample_multistep(batch_size, c, cfg_scale, num_steps)
    
    @torch.no_grad()
    def sample_onestep(self, batch_size, c=None, cfg_scale=None):
        """One-step generation: z_1 = z_0 + u(z_0, 0, 1)"""
        # Sample initial noise at t=0 (not t=1!)
        z_0 = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.device)
        
        # Create time tensors
        r = torch.zeros(batch_size, device=self.device)  # t=0 is noise
        t = torch.ones(batch_size, device=self.device)   # t=1 is data
        
        if self.use_cond and c is None:
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        # Predict average velocity from noise (t=0) to data (t=1)
        if cfg_scale is not None and cfg_scale > 0:
            u = self.student.forward_with_cfg(z_0, r, t, c, cfg_scale)
        else:
            u = self.student(z_0, r, t, c)
        
        # Forward update: data = noise + velocity
        z_1 = z_0 + u  # Note: plus, not minus!
        
        return z_1.clamp(-1, 1)

    @torch.no_grad()
    def sample_twostep(self, batch_size, c=None, cfg_scale=None):
        """Two-step generation with correct direction"""
        # Sample initial noise at t=0
        z_0 = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.device)
        
        if self.use_cond and c is None:
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        # Step 1: z_0 -> z_0.5 (from noise toward data)
        r1 = torch.zeros(batch_size, device=self.device)
        t1 = torch.full((batch_size,), 0.5, device=self.device)
        
        if cfg_scale is not None and cfg_scale > 0:
            u1 = self.student.forward_with_cfg(z_0, r1, t1, c, cfg_scale)
        else:
            u1 = self.student(z_0, r1, t1, c)
        
        z_05 = z_0 + 0.5 * u1  # Plus! Going from noise to data
        
        # Step 2: z_0.5 -> z_1 (continue toward data)
        r2 = torch.full((batch_size,), 0.5, device=self.device)
        t2 = torch.ones(batch_size, device=self.device)
        
        if cfg_scale is not None and cfg_scale > 0:
            u2 = self.student.forward_with_cfg(z_05, r2, t2, c, cfg_scale)
        else:
            u2 = self.student(z_05, r2, t2, c)
        
        z_1 = z_05 + 0.5 * u2  # Plus again!
        
        return z_1.clamp(-1, 1)
    
    @torch.no_grad()
    def sample_multistep(self, batch_size, c=None, cfg_scale=None, num_steps=4):
        """Multi-step generation for arbitrary number of steps"""
        # Sample initial noise
        z = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.device)
        
        # Generate class labels if needed
        if self.use_cond and c is None:
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        # Create time schedule (uniform for simplicity)
        t_schedule = torch.linspace(0, 1, num_steps + 1, device=self.device)
        
        # Multi-step sampling
        for i in range(num_steps):
            r = t_schedule[i]
            t = t_schedule[i + 1]
            
            r_batch = r.expand(batch_size)
            t_batch = t.expand(batch_size)
            
            if cfg_scale is not None and cfg_scale > 0:
                u = self.student.forward_with_cfg(z, r_batch, t_batch, c, cfg_scale)
            else:
                u = self.student(z, r_batch, t_batch, c)
            
            # Update z
            dt = (t - r).item()
            z = z + dt * u
        
        # Clip final output
        z = z.clamp(-1, 1)
        return z
    
    @torch.no_grad()
    def sample_each_class(self, n_per_class, num_steps=1, cfg_scale=None):
        """Sample from each class"""
        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        batch_size = self.num_classes * n_per_class
        
        return self.sample(batch_size=batch_size, class_labels=c, num_steps=num_steps, cfg_scale=cfg_scale)


def create_student_from_teacher(teacher_dit, teacher_checkpoint_path=None):
    """
    Create a SplitMeanFlow student model from a trained teacher DiT.
    """
    # Check if teacher has register tokens
    num_register_tokens = 0
    if hasattr(teacher_dit, 'register_tokens') and teacher_dit.register_tokens is not None:
        num_register_tokens = teacher_dit.register_tokens.shape[0]
    
    student = SMDiT(
        input_size=teacher_dit.x_embedder.img_size[0],
        patch_size=teacher_dit.patch_size,
        in_channels=teacher_dit.in_channels,
        dim=teacher_dit.t_embedder.mlp[-1].out_features,
        depth=len(teacher_dit.blocks),
        num_heads=teacher_dit.num_heads,
        num_register_tokens=num_register_tokens,
        class_dropout_prob=0.0,  # No dropout for student!
        num_classes=teacher_dit.num_classes,
        learn_sigma=teacher_dit.learn_sigma,
    )
    
    teacher_state = teacher_dit.state_dict()
    student_state = student.state_dict()
    
    for teacher_key, teacher_param in teacher_state.items():
        if 't_embedder' in teacher_key:
            # Map teacher t_embedder to student's single_embedder
            student_key = teacher_key.replace('t_embedder', 't_embedder.single_embedder')
            if student_key in student_state:
                print(f"Copying {teacher_key} -> {student_key}")
                student_state[student_key].copy_(teacher_param)
        elif teacher_key in student_state and teacher_param.shape == student_state[teacher_key].shape:
            # Copy other matching parameters
            print(f"Copying {teacher_key}")
            student_state[teacher_key].copy_(teacher_param)
    
    
    student.load_state_dict(student_state)
    
    print("\nVerifying weight transfer:")
    print(f"Teacher t_embedder.mlp[0].weight mean: {teacher_dit.t_embedder.mlp[0].weight.mean().item():.4f}")
    print(f"Student single_embedder.mlp[0].weight mean: {student.t_embedder.single_embedder.mlp[0].weight.mean().item():.4f}")
    
    return student