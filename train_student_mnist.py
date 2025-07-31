from dit import DiT
from splitmeanflow import SplitMeanFlowDiT, SplitMeanFlow, create_student_from_teacher
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


def save_grayscale_grid(samples, filename, nrow=10):
    """Convert grayscale to RGB for better visualization"""
    samples_rgb = samples.repeat(1, 3, 1, 1)
    grid = make_grid(samples_rgb, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, filename)


def main():
    # Training configuration
    n_steps = 500000  # Reduced for MNIST
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256  # Larger batch size for MNIST
    ema_decay = 0.9999
    
    # MNIST settings
    image_size = 28  # MNIST native size
    image_channels = 1  # MNIST is grayscale
    num_classes = 10  # 10 digits
    
    # SplitMeanFlow specific configs
    flow_ratio = 0.7  # Higher for stability as paper suggests
    teacher_cfg_scale = 5.0  # Fixed CFG scale for teacher guidance
    time_sampling = "logit_normal_cosine"  # Match teacher
    sigma_min = 1e-06  # Match teacher
    
    # Paths - Update to your MNIST teacher checkpoint
    teacher_checkpoint_path = "./mnist_teacher_step_99999.pth"
    checkpoint_root_path = '/mnt/nvme/checkpoint/splitmeanflow_mnist/'
    os.makedirs(checkpoint_root_path, exist_ok=True)
    os.makedirs('/mnt/nvme/images_student_mnist', exist_ok=True)

    resume_from_checkpoint = False
    resume_checkpoint_path = "/mnt/nvme/checkpoint/splitmeanflow_mnist/step_49999.pth"
    
    # Initialize Comet ML experiment
    experiment = Experiment(
        project_name="splitmeanflow-student-mnist",
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "dataset": "MNIST",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "model": "SplitMeanFlow-DiT-MNIST",
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
    
    # Dataset - MNIST with normalization to [-1, 1]
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # [0, 1]
        T.Normalize((0.5,), (0.5,)),  # Convert to [-1, 1] for single channel
    ])
    
    dataset = torchvision.datasets.MNIST(
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
    
    # Load teacher model with MNIST architecture
    print("Loading MNIST teacher model...")
    teacher_dit = DiT(
        input_size=image_size,
        patch_size=4,  # MNIST uses larger patches
        in_channels=image_channels,
        dim=256,  # MNIST teacher uses smaller dim
        depth=8,  # MNIST teacher uses fewer layers
        num_heads=4,  # MNIST teacher uses fewer heads
        num_classes=num_classes,
        learn_sigma=False,
        class_dropout_prob=0.1,
    ).to(device)
    
    # Load teacher checkpoint
    checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
  
    teacher_dit.load_state_dict(checkpoint['model'])
    print("Loaded teacher model weights")
    
    # Create teacher RectifiedFlow wrapper
    teacher_model = RectifiedFlow(
        net=teacher_dit,
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        use_logit_normal_cosine=True,  # Match teacher training
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
        use_immiscible=True,  # Use immiscible sampling like teacher
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
    
    # Optimizer with lower weight decay for MNIST
    optimizer = optim.Adam(student_net.parameters(), lr=1e-4, weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler()

    # Resume from checkpoint if requested
    start_step = 0
    if resume_from_checkpoint:
        # Find the latest checkpoint if no specific path provided
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
                
                # Save with grayscale handling
                img_save_path = f"/mnt/nvme/images_student_mnist/step{step}_{num_steps}step.png"
                save_grayscale_grid(samples, img_save_path)
                
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
                
                img_save_path_ema = f"/mnt/nvme/images_student_mnist/step{step}_{num_steps}step_ema.png"
                save_grayscale_grid(samples_ema, img_save_path_ema)
                
                experiment.log_image(
                    img_save_path_ema,
                    name=f"{num_steps}step_generation_ema",
                    step=step
                )
        
        # Also generate specific digits with 1-step
        print("Generating specific digits with 1-step...")
        with torch.no_grad():
            # Generate 50 samples of digit '8'
            digit_8_labels = torch.full((50,), 8, device=device)
            digit_8_samples = splitmeanflow_ema.sample(
                class_labels=digit_8_labels,
                num_steps=1,
                cfg_scale=None
            )
            
            img_save_path_digit = f"/mnt/nvme/images_student_mnist/step{step}_digit8_1step.png"
            save_grayscale_grid(digit_8_samples, img_save_path_digit, nrow=10)
            
            experiment.log_image(
                img_save_path_digit,
                name="digit_8_1step",
                step=step
            )
    
    # Training loop
    losses = []
    boundary_losses = []
    consistency_losses = []
    gradient_clip = 1.0

    initial_dataloader_steps = start_step
    # for _ in range(initial_dataloader_steps):
    #     _ = next(train_dataloader)
    
    with tqdm(range(start_step, n_steps), initial=start_step, total=n_steps, dynamic_ncols=True) as pbar:

        pbar.set_description("Training SplitMeanFlow MNIST")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            
            # Get images and labels
            x1 = data[0].to(device)  # Already in [-1, 1] due to normalization
            y = data[1].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Training returns loss and loss type
                loss, loss_type = splitmeanflow(x1, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            update_ema(student_ema, student_net, decay=ema_decay)
            
            # Track losses
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
            
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                experiment.log_metric("avg_loss_100", avg_loss, step=step)
            
            if step % 1000 == 0 or step == n_steps - 1:  # More frequent for MNIST
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
            
            if step % 2500 == 0 or step == n_steps - 1:  # More frequent checkpoints
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_root_path, f"step_{step}.pth")
                state_dict = {
                    "model": student_net.state_dict(),
                    "ema": student_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {
                        "dataset": "MNIST",
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
    checkpoint_path = os.path.join(checkpoint_root_path, "student_mnist_final.pth")
    state_dict = {
        "model": student_net.state_dict(),
        "ema": student_ema.state_dict(),
        "config": {
            "dataset": "MNIST",
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