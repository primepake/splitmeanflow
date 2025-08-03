from dit import DiT
from splitmeanflow import SplitMeanFlow, create_student_from_teacher
from model import RectifiedFlow
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
import torch.optim as optim
import os
import moviepy.editor as mpy
from comet_ml import Experiment
import torch.nn.functional as F

# Clean EMA functions (same as teacher)
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def run_diagnostics(splitmeanflow, step, experiment):
    """Run diagnostic tests on the model"""
    device = splitmeanflow.device
    batch_size = 16
    
    # Test 1: Boundary condition accuracy
    t_test = torch.rand(batch_size, device=device) * 0.8 + 0.1
    x_test = torch.randn(batch_size, splitmeanflow.channels, 
                         splitmeanflow.image_size, splitmeanflow.image_size, device=device)
    z_test = torch.randn_like(x_test)
    
    # Create class labels for CIFAR-10
    y_test = torch.randint(0, splitmeanflow.num_classes, (batch_size,), device=device)
    
    # Create z_t
    if splitmeanflow.scheduler is not None:
        alpha_t, sigma_t = splitmeanflow.scheduler.get_cosine_schedule_params(t_test, splitmeanflow.sigma_min)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
    else:
        t_expand = t_test.view(-1, 1, 1, 1)
        alpha_t = t_expand
        sigma_t = 1 - (1 - splitmeanflow.sigma_min) * t_expand
    
    z_t = sigma_t * z_test + alpha_t * x_test
    
    # Teacher prediction
    with torch.no_grad():
        v_teacher = splitmeanflow.teacher.net(z_t, t_test, y_test)  # Pass y_test
    
    # Student prediction for boundary condition
    u_student = splitmeanflow.student(z_t, t_test, t_test, y_test)  # Pass y_test
    
    boundary_error = F.mse_loss(u_student, v_teacher).item()
    
    # Test 2: Interval consistency
    r = torch.zeros(batch_size, device=device)
    t = torch.ones(batch_size, device=device)
    s = torch.full((batch_size,), 0.5, device=device)
    
    u_full = splitmeanflow.student(z_t, r, t, y_test)
    u_first = splitmeanflow.student(z_t, r, s, y_test)
    u_second = splitmeanflow.student(z_t, s, t, y_test)
    
    consistency_target = 0.5 * u_first + 0.5 * u_second
    consistency_error = F.mse_loss(u_full, consistency_target).item()
    
    # Test 3: Check specific intervals used in sampling
    # Test [0, 1] interval
    z_0 = torch.randn_like(x_test)
    u_01 = splitmeanflow.student(z_0, r, t, y_test)
    
    # Test [0, 0.5] and [0.5, 1] intervals
    u_05_first = splitmeanflow.student(z_0, r, s, y_test)
    z_05 = z_0 + 0.5 * u_05_first
    u_05_second = splitmeanflow.student(z_05, s, t, y_test)
    
    # Log metrics
    experiment.log_metric("diagnostic_boundary_error", boundary_error, step=step)
    experiment.log_metric("diagnostic_consistency_error", consistency_error, step=step)
    experiment.log_metric("diagnostic_velocity_magnitude", u_full.abs().mean().item(), step=step)
    experiment.log_metric("diagnostic_u01_magnitude", u_01.abs().mean().item(), step=step)
    
    print(f"\n=== Diagnostics at step {step} ===")
    print(f"Boundary error: {boundary_error:.6f}")
    print(f"Consistency error: {consistency_error:.6f}")
    print(f"Velocity magnitude: {u_full.abs().mean().item():.4f}")
    print(f"[0,1] velocity magnitude: {u_01.abs().mean().item():.4f}")
    print("="*30)


def main():
    # Training configuration
    n_steps = 500000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    ema_decay = 0.9999
    
    # Image settings
    image_size = 32
    image_channels = 3
    num_classes = 10
    
    # SplitMeanFlow specific configs
    flow_ratio = 1.0
    teacher_cfg_scale = 5.0
    time_sampling = "uniform"
    sigma_min = 1e-06
    lr = 1e-4
    
    # Paths
    teacher_checkpoint_path = "./cifar_200000.pth"
    checkpoint_root_path = '/checkpoint/splitmeanflow_cifar10/'
    os.makedirs(checkpoint_root_path, exist_ok=True)
    os.makedirs('/mnt/nvme/images_student_cifar10', exist_ok=True)

    resume_from_checkpoint = False
    resume_checkpoint_path = None  # Will auto-find latest if None
    
    experiment = Experiment(
        project_name="splitmeanflow-student-cifar10",
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "dataset": "CIFAR-10",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": lr,
        "model": "SplitMeanFlow-DiT-CIFAR10",
        "teacher_checkpoint": teacher_checkpoint_path,
        "flow_ratio": flow_ratio,
        "teacher_cfg_scale": teacher_cfg_scale,
        "time_sampling": time_sampling,
        "sigma_min": sigma_min,
        "ema_decay": ema_decay,
        "image_size": image_size,
        "image_channels": image_channels,
        "num_classes": num_classes,
        "optimizer": "Adam",
        "mixed_precision": "bfloat16",
        "weight_decay": 0.0001,
    })
    
    transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])

    dataset = torchvision.datasets.CIFAR10(
        root="/mnt/nvme/",
        train=True,
        download=True,
        transform=transform,
    )
    
    def cycle(iterable):
        while True:
            for i in iterable:
                yield i
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    train_dataloader = cycle(train_dataloader)
    
    # Load teacher model
    print("Loading CIFAR-10 teacher model...")
    teacher_dit = DiT(
        input_size=image_size,
        patch_size=2,
        in_channels=image_channels,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
        learn_sigma=False,
        class_dropout_prob=0.1,
    ).to(device)
    
    checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    
    teacher_dit.load_state_dict(checkpoint['ema_model'])
    print("Loaded teacher ema weights")
    
    # Create teacher RectifiedFlow wrapper
    teacher_model = RectifiedFlow(
        net=teacher_dit,
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        use_logit_normal_cosine=True,  # Added to match MNIST
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
        timestep_min=1e-8,
        timestep_max=1.0-1e-8,
    )
    teacher_model.eval()
    
    # Create student model
    print("Creating student model from teacher...")
    student_net = create_student_from_teacher(teacher_dit).to(device)
    
    # Create student EMA model
    print("Creating student EMA model...")
    student_ema = deepcopy(student_net).to(device)
    requires_grad(student_ema, False)
    student_ema.eval()
    update_ema(student_ema, student_net, decay=0)  # Full copy
    
    # Create SplitMeanFlow trainer
    splitmeanflow = SplitMeanFlow(
        student_net=student_net,
        teacher_model=teacher_model,
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        flow_ratio=flow_ratio,
        cfg_scale=teacher_cfg_scale,
        time_sampling=time_sampling,
        sigma_min=sigma_min,
        use_immiscible=True,  # Added to match MNIST
    )
    
    # Create SplitMeanFlow wrapper for EMA model (for sampling)
    splitmeanflow_ema = SplitMeanFlow(
        student_net=student_ema,
        teacher_model=teacher_model,
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        flow_ratio=flow_ratio,
        cfg_scale=teacher_cfg_scale,
        time_sampling=time_sampling,
        sigma_min=sigma_min,
        use_immiscible=True,
    )
    
    optimizer = optim.Adam(student_net.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    start_step = 0
    if resume_from_checkpoint:
        if resume_checkpoint_path is None:
            checkpoint_files = [f for f in os.listdir(checkpoint_root_path) if f.startswith('step_') and f.endswith('.pth')]
            if checkpoint_files:
                # Sort by step number
                checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                resume_checkpoint_path = os.path.join(checkpoint_root_path, checkpoint_files[-1])
                print(f"Found latest checkpoint: {resume_checkpoint_path}")
        
        # Load checkpoint if it exists
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            print(f"Resuming from checkpoint: {resume_checkpoint_path}")
            try:
                resume_checkpoint = torch.load(resume_checkpoint_path, map_location=device)
                
                # Validate checkpoint
                required_keys = ['model', 'ema', 'optimizer', 'step']
                missing_keys = [k for k in required_keys if k not in resume_checkpoint]
                if missing_keys:
                    print(f"Warning: Checkpoint missing keys: {missing_keys}")
                    print("Starting from scratch instead")
                else:
                    # Load model states
                    student_net.load_state_dict(resume_checkpoint['model'])
                    student_ema.load_state_dict(resume_checkpoint['ema'])
                    optimizer.load_state_dict(resume_checkpoint['optimizer'])
                    
                    # Get the starting step
                    start_step = resume_checkpoint['step'] + 1
                    
                    # Move optimizer state to device
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(device)
                    
                    print(f"Successfully resumed from step {start_step}")
                    
                    # Log checkpoint info
                    if 'config' in resume_checkpoint:
                        print("Checkpoint config:")
                        for k, v in resume_checkpoint['config'].items():
                            print(f"  - {k}: {v}")
                    
                    # Log to experiment
                    experiment.log_metric("resumed_from_step", start_step, step=start_step)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch")
                start_step = 0
        else:
            print("No checkpoint found to resume from, starting from scratch")
    
    def sample_and_log_images():
        """Sample images from both regular and EMA models"""
        for num_steps in [1, 2, 4]:  # Test different step counts
            print(f"Sampling {num_steps}-step images at step {step}...")
            
            # Sample from regular model
            student_net.eval()
            with torch.no_grad():
                samples = splitmeanflow.sample_each_class(
                    n_per_class=10,
                    num_steps=num_steps,
                    cfg_scale=None  # No CFG at inference for student
                )
                
                # Convert from [-1, 1] to [0, 1] for visualization
                samples = (samples + 1) / 2
                
                log_img = make_grid(samples, nrow=10)
                img_save_path = f"/mnt/nvme/images_student_cifar10/step{step}_{num_steps}step.png"
                save_image(log_img, img_save_path)
                
                experiment.log_image(
                    img_save_path,
                    name=f"{num_steps}step_generation",
                    step=step
                )
            student_net.train()
            
            # Sample from EMA model
            print(f"Sampling {num_steps}-step images from EMA model...")
            with torch.no_grad():
                samples_ema = splitmeanflow_ema.sample_each_class(
                    n_per_class=10,
                    num_steps=num_steps,
                    cfg_scale=None
                )
                
                # Convert from [-1, 1] to [0, 1] for visualization
                samples_ema = (samples_ema + 1) / 2
                
                log_img_ema = make_grid(samples_ema, nrow=10)
                img_save_path_ema = f"/mnt/nvme/images_student_cifar10/step{step}_{num_steps}step_ema.png"
                save_image(log_img_ema, img_save_path_ema)
                
                experiment.log_image(
                    img_save_path_ema,
                    name=f"{num_steps}step_generation_ema",
                    step=step
                )
        
        # Also generate specific classes with 1-step
        print("Generating specific classes with 1-step...")
        with torch.no_grad():
            # Generate 50 samples of class 'airplane' (class 0 in CIFAR-10)
            airplane_labels = torch.full((50,), 0, device=device)
            airplane_samples = splitmeanflow_ema.sample(
                class_labels=airplane_labels,
                num_steps=1,
                cfg_scale=None
            )
            
            # Convert from [-1, 1] to [0, 1]
            airplane_samples = (airplane_samples + 1) / 2
            
            img_save_path_class = f"/mnt/nvme/images_student_cifar10/step{step}_airplane_1step.png"
            save_image(make_grid(airplane_samples, nrow=10), img_save_path_class)
            
            experiment.log_image(
                img_save_path_class,
                name="airplane_1step",
                step=step
            )
    
    # Training loop
    losses = []
    boundary_losses = []
    consistency_losses = []
    gradient_clip = 1.0

    initial_dataloader_steps = start_step
    
    with tqdm(range(start_step, n_steps), initial=start_step, total=n_steps, dynamic_ncols=True) as pbar:
        pbar.set_description("Training SplitMeanFlow CIFAR-10")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            
            x1 = data[0].to(device)
            x1 = x1 * 2 - 1
            y = data[1].to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, loss_type = splitmeanflow(x1, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            
            update_ema(student_ema, student_net, decay=ema_decay)
            
            losses.append(loss.item())
            if loss_type == "boundary":
                boundary_losses.append(loss.item())
            else:
                consistency_losses.append(loss.item())
            
            pbar.set_postfix({
                "loss": loss.item(), 
                "type": loss_type,
                "grad": grad_norm.item()
            })
            
            experiment.log_metric("loss", loss.item(), step=step)
            experiment.log_metric(f"{loss_type}_loss", loss.item(), step=step)
            experiment.log_metric("grad_norm", grad_norm.item(), step=step)
            
            # Run diagnostics
            if step % 5000 == 0:
                run_diagnostics(splitmeanflow, step, experiment)
            
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                experiment.log_metric("avg_loss_100", avg_loss, step=step)
            
            if step % 2000 == 0 or step == n_steps - 1:  # More frequent than original
                avg_loss = sum(losses) / len(losses) if losses else 0
                avg_boundary = sum(boundary_losses) / len(boundary_losses) if boundary_losses else 0
                avg_consistency = sum(consistency_losses) / len(consistency_losses) if consistency_losses else 0
                
                print(f"\nStep: {step+1}/{n_steps}")
                print(f"Average loss: {avg_loss:.4f}")
                print(f"Boundary loss: {avg_boundary:.4f} (n={len(boundary_losses)})")
                print(f"Consistency loss: {avg_consistency:.4f} (n={len(consistency_losses)})")
                
                losses.clear()
                boundary_losses.clear()
                consistency_losses.clear()
                
                sample_and_log_images()
            
            if step % 5000 == 0 or step == n_steps - 1:  # Keep same checkpoint frequency
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_root_path, f"step_{step}.pth")
                state_dict = {
                    "model": student_net.state_dict(),
                    "ema": student_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {
                        "dataset": "CIFAR-10",
                        "image_size": image_size,
                        "image_channels": image_channels,
                        "ema_decay": ema_decay,
                        "sigma_min": sigma_min,
                        "flow_ratio": flow_ratio,
                        "teacher_cfg_scale": teacher_cfg_scale,
                        "time_sampling": time_sampling,
                        "gradient_clip": gradient_clip,
                    }
                }
                torch.save(state_dict, checkpoint_path)
                
                experiment.log_model(
                    name=f"checkpoint_step_{step}",
                    file_or_folder=checkpoint_path
                )
    
    # Final save
    checkpoint_path = os.path.join(checkpoint_root_path, "student_cifar10_final.pth")
    state_dict = {
        "model": student_net.state_dict(),
        "ema": student_ema.state_dict(),
        "config": {
            "dataset": "CIFAR-10",
            "image_size": image_size,
            "image_channels": image_channels,
            "ema_decay": ema_decay,
            "sigma_min": sigma_min,
            "flow_ratio": flow_ratio,
            "teacher_cfg_scale": teacher_cfg_scale,
            "time_sampling": time_sampling,
            "gradient_clip": gradient_clip,
        }
    }
    torch.save(state_dict, checkpoint_path)
    
    experiment.log_model(
        name="final_model",
        file_or_folder=checkpoint_path
    )
    
    experiment.end()
    print("Training completed!")
    
    # Print summary
    total_params = sum(p.numel() for p in student_net.parameters())
    teacher_params = sum(p.numel() for p in teacher_dit.parameters())
    print(f"\nModel Summary:")
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {total_params:,}")
    print(f"Parameter reduction: {(1 - total_params/teacher_params)*100:.1f}%")


if __name__ == "__main__":
    main()