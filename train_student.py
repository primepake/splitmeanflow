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


def main():
    # Training configuration
    n_steps = 100000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    ema_decay = 0.9999
    
    # Image settings (matching teacher)
    image_size = 16
    image_channels = 3
    
    # SplitMeanFlow specific configs
    flow_ratio = 0.7  # Higher for stability as paper suggests
    teacher_cfg_scale = 3.0  # Fixed CFG scale for teacher guidance
    time_sampling = "cosine"  # Match teacher
    sigma_min = 1e-06  # Match teacher
    
    # Paths
    teacher_checkpoint_path = "/mnt/nvme/checkpoint/dit_direct/model_direct_final.pth"
    checkpoint_root_path = '/mnt/nvme/checkpoint/splitmeanflow/'
    os.makedirs(checkpoint_root_path, exist_ok=True)
    os.makedirs('/mnt/nvme/images_student', exist_ok=True)
    
    # Initialize Comet ML experiment
    experiment = Experiment(
        project_name="splitmeanflow-student",
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "model": "SplitMeanFlow-DiT",
        "teacher_checkpoint": teacher_checkpoint_path,
        "flow_ratio": flow_ratio,
        "teacher_cfg_scale": teacher_cfg_scale,
        "time_sampling": time_sampling,
        "sigma_min": sigma_min,
        "ema_decay": ema_decay,
        "image_size": image_size,
        "optimizer": "Adam",
        "mixed_precision": "bfloat16",
    })
    
    # Dataset (same as teacher)
    dataset = torchvision.datasets.CIFAR10(
        root="/mnt/nvme/",
        train=True,
        download=True,
        transform=T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),  # [0, 1]
        ]),
    )
    
    def cycle(iterable):
        while True:
            for i in iterable:
                yield i
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8
    )
    train_dataloader = cycle(train_dataloader)
    
    # Load teacher model
    print("Loading teacher model...")
    teacher_dit = DiT(
        input_size=image_size,
        patch_size=2,
        in_channels=image_channels,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        learn_sigma=False,
        class_dropout_prob=0.1,
    ).to(device)
    
    # Load teacher checkpoint
    checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    if 'ema' in checkpoint:
        teacher_dit.load_state_dict(checkpoint['ema'])
        print("Loaded teacher EMA weights")
    else:
        teacher_dit.load_state_dict(checkpoint['model'])
        print("Loaded teacher model weights")
    
    # Create teacher RectifiedFlow wrapper
    teacher_model = RectifiedFlow(
        net=teacher_dit,
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=10,
    )
    teacher_model.eval()
    
    # Create student model
    print("Creating student model from teacher...")
    student_net = create_student_from_teacher(teacher_dit).to(device)
    
    # Create student EMA model (same as teacher)
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
        num_classes=10,
        flow_ratio=flow_ratio,
        cfg_scale=teacher_cfg_scale,
        time_sampling=time_sampling,
        sigma_min=sigma_min,
    )
    
    # Create SplitMeanFlow wrapper for EMA model (for sampling)
    splitmeanflow_ema = SplitMeanFlow(
        student_net=student_ema,
        teacher_model=teacher_model,
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=10,
        flow_ratio=flow_ratio,
        cfg_scale=teacher_cfg_scale,
        time_sampling=time_sampling,
        sigma_min=sigma_min,
    )
    
    # Optimizer (same as teacher)
    optimizer = optim.Adam(student_net.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    def sample_and_log_images():
        """Sample images from both regular and EMA models"""
        for num_steps in [1, 2]:
            print(f"Sampling {num_steps}-step images at step {step}...")
            
            # Sample from regular model
            student_net.eval()
            with torch.no_grad():
                samples, = splitmeanflow.sample_each_class(
                    n_per_class=10,
                    num_steps=num_steps,
                    cfg_scale=None  # No CFG at inference
                )
                
                log_img = make_grid(samples, nrow=10)
                img_save_path = f"/mnt/nvme/images_student/step{step}_{num_steps}step.png"
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
                
                log_img_ema = make_grid(samples_ema, nrow=10)
                img_save_path_ema = f"/mnt/nvme/images_student/step{step}_{num_steps}step_ema.png"
                save_image(log_img_ema, img_save_path_ema)
                
                experiment.log_image(
                    img_save_path_ema,
                    name=f"{num_steps}step_generation_ema",
                    step=step
                )
    
    # Training loop
    losses = []
    boundary_losses = []
    consistency_losses = []
    
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training SplitMeanFlow")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            
            # Get images and labels
            x1 = data[0].to(device)  # Already in [0, 1]
            y = data[1].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Modified to return loss type
                loss, loss_type = splitmeanflow(x1, y)
            
            scaler.scale(loss).backward()
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
            
            pbar.set_postfix({"loss": loss.item(), "type": loss_type})
            experiment.log_metric("loss", loss.item(), step=step)
            experiment.log_metric(f"{loss_type}_loss", loss.item(), step=step)
            
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                experiment.log_metric("avg_loss_100", avg_loss, step=step)
            
            if step % 2000 == 0 or step == n_steps - 1:
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
            
            if step % 5000 == 0 or step == n_steps - 1:
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_root_path, f"step_{step}.pth")
                state_dict = {
                    "model": student_net.state_dict(),
                    "ema": student_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {
                        "image_size": image_size,
                        "image_channels": image_channels,
                        "ema_decay": ema_decay,
                        "sigma_min": sigma_min,
                        "flow_ratio": flow_ratio,
                        "teacher_cfg_scale": teacher_cfg_scale,
                        "time_sampling": time_sampling,
                    }
                }
                torch.save(state_dict, checkpoint_path)
                
                experiment.log_model(
                    name=f"checkpoint_step_{step}",
                    file_or_folder=checkpoint_path
                )
    
    # Final save
    checkpoint_path = os.path.join(checkpoint_root_path, "student_final.pth")
    state_dict = {
        "model": student_net.state_dict(),
        "ema": student_ema.state_dict(),
        "config": {
            "image_size": image_size,
            "image_channels": image_channels,
            "ema_decay": ema_decay,
            "sigma_min": sigma_min,
            "flow_ratio": flow_ratio,
            "teacher_cfg_scale": teacher_cfg_scale,
            "time_sampling": time_sampling,
        }
    }
    torch.save(state_dict, checkpoint_path)
    
    experiment.log_model(
        name="final_model",
        file_or_folder=checkpoint_path
    )
    
    experiment.end()


if __name__ == "__main__":
    main()