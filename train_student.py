from dit import DiT
from splitmeanflow import SplitMeanFlowDiT, SplitMeanFlow, create_student_from_teacher
from model import RectifiedFlow
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from ema import LitEma
from bitsandbytes.optim import AdamW8bit
import os

import moviepy.editor as mpy
from comet_ml import Experiment


def main():
    # Training configuration
    n_steps = 100000  # Fewer steps needed for student
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    teacher_checkpoint_path = "model.pth"  # Path to trained teacher
    
    # SplitMeanFlow specific configs
    flow_ratio = 0.5  # 50% boundary condition, 50% interval splitting
    cfg_scale = 3.0   # Fixed CFG scale for teacher guidance
    
    # Create directories
    os.makedirs('images_student', exist_ok=True)
    os.makedirs('checkpoints_student', exist_ok=True)

    # Dataset
    dataset = torchvision.datasets.CIFAR10(
        root="/mnt/g",
        train=True,
        download=True,
        transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
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
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        learn_sigma=False,
        class_dropout_prob=0.1,  # Teacher has dropout
    ).to(device)
    
    teacher_model = RectifiedFlow(
        net=teacher_dit,
        device=device,
        channels=3,
        image_size=32,
        num_classes=10,
    )
    
    # Load teacher checkpoint
    checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    teacher_model.net.load_state_dict(checkpoint['model'])
    if 'ema' in checkpoint:
        # Use EMA weights if available
        teacher_ema = LitEma(teacher_model.net)
        teacher_ema.load_state_dict(checkpoint['ema'])
        teacher_ema.copy_to(teacher_model.net)
    print("Teacher model loaded successfully!")
    
    # Create student model
    print("Creating student model from teacher...")
    student_net = create_student_from_teacher(teacher_model).to(device)
    
    # Create SplitMeanFlow trainer
    splitmeanflow = SplitMeanFlow(
        student_net=student_net,
        teacher_model=teacher_model,
        device=device,
        channels=3,
        image_size=32,
        num_classes=10,
        flow_ratio=flow_ratio,
        cfg_scale=cfg_scale,
        time_sampling="uniform",  # or "lognormal"
    )
    
    # Student EMA
    student_ema = LitEma(student_net)
    
    # Optimizer - lower learning rate for student
    optimizer = AdamW8bit(student_net.parameters(), lr=5e-5, weight_decay=0.0)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Logging with Comet ML
    experiment = Experiment(
        api_key="YOUR_API_KEY",  # Replace with your Comet ML API key
        project_name="splitmeanflow-student",
        workspace="YOUR_WORKSPACE",  # Replace with your workspace
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "flow_ratio": flow_ratio,
        "cfg_scale": cfg_scale,
        "batch_size": batch_size,
        "learning_rate": 5e-5,
        "teacher_checkpoint": teacher_checkpoint_path,
        "n_steps": n_steps,
        "student_architecture": "DiT",
        "time_sampling": "uniform"
    })
    
    def sample_and_log_images(step):
        """Sample images using 1-step and 2-step generation"""
        
        # Test different sampling strategies
        for num_steps in [1, 2]:
            print(f"Sampling with {num_steps} step(s) at training step {step}...")
            
            # Sample without EMA
            student_net.eval()
            with torch.no_grad():
                samples = splitmeanflow.sample_each_class(
                    n_per_class=10, 
                    num_steps=num_steps,
                    cfg_scale=None  # No CFG needed at inference!
                )
            
            log_img = make_grid(samples, nrow=10)
            img_save_path = f"images_student/step{step}_{num_steps}step.png"
            save_image(log_img, img_save_path)
            
            # Log to Comet ML
            experiment.log_image(
                img_save_path, 
                name=f"{num_steps}-step_generation",
                step=step
            )
            
            # Sample with EMA
            print(f"Sampling with EMA, {num_steps} step(s)...")
            student_ema.store(student_net.parameters())
            student_ema.copy_to(student_net)
            
            with torch.no_grad():
                samples_ema = splitmeanflow.sample_each_class(
                    n_per_class=10,
                    num_steps=num_steps,
                    cfg_scale=None
                )
            
            log_img_ema = make_grid(samples_ema, nrow=10)
            img_save_path_ema = f"images_student/step{step}_{num_steps}step_ema.png"
            save_image(log_img_ema, img_save_path_ema)
            
            # Log EMA image to Comet ML
            experiment.log_image(
                img_save_path_ema,
                name=f"{num_steps}-step_generation_EMA",
                step=step
            )
            
            student_ema.restore(student_net.parameters())
            student_net.train()
    
    # Training loop
    losses = []
    boundary_losses = []
    consistency_losses = []
    
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training SplitMeanFlow")
        student_net.train()
        
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            
            x = data[0].to(device)
            y = data[1].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = splitmeanflow(x, y)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            student_ema(student_net)
            
            # Logging
            if not torch.isnan(loss):
                losses.append(loss.item())
                
                # Track which type of loss it was (approximate)
                if torch.rand(1).item() < flow_ratio:
                    boundary_losses.append(loss.item())
                else:
                    consistency_losses.append(loss.item())
                
                pbar.set_postfix({"loss": loss.item()})
                experiment.log_metric("loss", loss.item(), step=step)
            
            # Periodic logging
            if step % 5000 == 0 and step > 0:
                avg_loss = sum(losses) / len(losses) if losses else 0
                avg_boundary = sum(boundary_losses) / len(boundary_losses) if boundary_losses else 0
                avg_consistency = sum(consistency_losses) / len(consistency_losses) if consistency_losses else 0
                
                print(f"\nStep: {step}/{n_steps}")
                print(f"Average loss: {avg_loss:.4f}")
                print(f"Boundary loss: {avg_boundary:.4f}")
                print(f"Consistency loss: {avg_consistency:.4f}")
                
                # Log metrics to Comet ML
                experiment.log_metrics({
                    "avg_loss": avg_loss,
                    "boundary_loss": avg_boundary,
                    "consistency_loss": avg_consistency
                }, step=step)
                
                losses.clear()
                boundary_losses.clear()
                consistency_losses.clear()
            
            # Sample images
            if step % 10000 == 0 or step == n_steps - 1:
                sample_and_log_images(step)
            
            # Save checkpoint
            if step % 25000 == 0 or step == n_steps - 1:
                print(f"Saving checkpoint at step {step}...")
                state_dict = {
                    "student": student_net.state_dict(),
                    "student_ema": student_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {
                        "flow_ratio": flow_ratio,
                        "cfg_scale": cfg_scale,
                    }
                }
                torch.save(state_dict, f"checkpoints_student/student_step{step}.pth")
                
                # Log model checkpoint to Comet ML
                experiment.log_model(
                    name=f"student_checkpoint_step_{step}",
                    file_or_folder=f"checkpoints_student/student_step{step}.pth"
                )
    
    # Final save
    print("Training completed! Saving final checkpoint...")
    final_state = {
        "student": student_net.state_dict(),
        "student_ema": student_ema.state_dict(),
        "config": {
            "flow_ratio": flow_ratio,
            "cfg_scale": cfg_scale,
        }
    }
    torch.save(final_state, "checkpoints_student/student_final.pth")
    
    # Log final model to Comet ML
    experiment.log_model(
        name="student_final_model",
        file_or_folder="checkpoints_student/student_final.pth"
    )
    
    # Log final samples with different step counts
    print("Generating final samples...")
    sample_and_log_images(n_steps)
    
    # End the experiment
    experiment.end()


if __name__ == "__main__":
    main()