from dit import DiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict

from model import RectifiedFlow, LogitNormalCosineScheduler
from fid_evaluation import FIDEvaluation

import moviepy.editor as mpy
from comet_ml import Experiment
import os
import torch.optim as optim
import torch.nn.functional as F
import glob

def find_latest_checkpoint(checkpoint_root_path):
    """Find the latest checkpoint in the directory"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_root_path, "step_*.pth"))
    if not checkpoint_files:
        return None
    
    # Extract step numbers and find the latest
    steps = []
    for f in checkpoint_files:
        try:
            step = int(os.path.basename(f).replace("step_", "").replace(".pth", ""))
            steps.append((step, f))
        except:
            continue
    
    if not steps:
        return None
    
    # Return the checkpoint with the highest step number
    latest_step, latest_checkpoint = max(steps, key=lambda x: x[0])
    return latest_checkpoint, latest_step


def main():
    n_steps = 400000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    
    image_size = 32
    image_channels = 3
    num_classes = 10
    
    # Create directories
    os.makedirs('/mnt/nvme/images_cifar10', exist_ok=True)
    os.makedirs('/mnt/nvme/results_cifar10', exist_ok=True)
    checkpoint_root_path = '/checkpoint/dit_cifar10/'
    os.makedirs(checkpoint_root_path, exist_ok=True)

    # Check for existing checkpoints
    checkpoint_info = find_latest_checkpoint(checkpoint_root_path)
    resume_from_checkpoint = checkpoint_info is not None
    start_step = 0

    # Initialize Comet ML experiment
    experiment = Experiment(
        project_name="dit-flow-matching-cifar10",
    )
    
   
    experiment.log_parameters({
        "dataset": "CIFAR-10",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "model": "DiT-CIFAR10",
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "patch_size": 2,
        "dropout_prob": 0.1,
        "optimizer": "Adam",
        "mixed_precision": "bfloat16",
        "fid_subset_size": 1000,
        "image_size": image_size,
        "image_channels": image_channels,
        "training_cfg_rate": 0.2,
        "lambda_weight": 0.05,
        "sigma_min": 1e-06,
        "ema_decay": 0.9999,
    })

    transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
    
    
    dataset = torchvision.datasets.CIFAR10(
        root="/mnt/nvme/",
        train=True,
        download=True,
        transform=transform,
    )

   
    timestep_scheduler = LogitNormalCosineScheduler(
        loc=0.0,        
        scale=1.0,      
        min_t=1e-8,    
        max_t=1.0-1e-8 
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
        num_workers=8
    )
    train_dataloader = cycle(train_dataloader)

    # Create model for CIFAR-10 generation
    model = DiT(
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
    
    # Create EMA model
    ema_model = deepcopy(model).eval()
    ema_decay = 0.9999
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load checkpoint if available
    if resume_from_checkpoint:
        checkpoint_path, start_step = checkpoint_info
        print(f"Resuming from checkpoint: {checkpoint_path} at step {start_step}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model states
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Update start step (checkpoint saves the current step, so we start from the next one)
        start_step = checkpoint['step'] + 1
        
        # Log checkpoint info
        experiment.log_metric("resumed_from_step", start_step, step=start_step)
        experiment.log_other("resumed_checkpoint_path", checkpoint_path)
        
        # Load config if needed (optional - can be used to verify consistency)
        if 'config' in checkpoint:
            loaded_config = checkpoint['config']
            print(f"Loaded configuration: {loaded_config}")
    
    # Create sampler (use EMA model for sampling)
    sampler = RectifiedFlow(
        ema_model,  # Use EMA model
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        use_logit_normal_cosine=True,
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
        timestep_min=1e-8,
        timestep_max=1.0-1e-8,
    )
    
    scaler = torch.cuda.amp.GradScaler()

    
    fid_subset_size = 1000
    
    def limited_cycle(iterable, max_batches):
        batch_count = 0
        while batch_count < max_batches:
            for i in iterable:
                yield i
                batch_count += 1
                if batch_count >= max_batches:
                    break
    
    fid_batches_needed = (fid_subset_size + batch_size - 1) // batch_size
    
    fid_dataset = torchvision.datasets.CIFAR10(
        root="/mnt/nvme",
        train=True,
        download=False,
        transform=transform,
    )
    
    fid_dataloader = torch.utils.data.DataLoader(
        fid_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False, 
        num_workers=4
    )
    
    fid_dataloader_limited = limited_cycle(fid_dataloader, fid_batches_needed)
    fid_eval = FIDEvaluation(batch_size, fid_dataloader_limited, sampler, num_fid_samples=100)
    
    def update_ema(ema_model, model, decay):
        """Update EMA model parameters"""
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.lerp_(p.data, 1 - decay)
    
    def sample_and_log_images():
        """Sample images from model"""
        for cfg_scale in [1.0, 2.5, 5.0]:
            print(f"Sampling images at step {step} with cfg_scale {cfg_scale}...")
            
          
            ema_model.eval()
            with torch.no_grad():
                
                samples, trajectory = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
                
                log_img = make_grid(samples, nrow=10, normalize=True, value_range=(-1, 1))
                img_save_path = f"/mnt/nvme/images_cifar10/step{step}_cfg{cfg_scale}.png"
                save_image(log_img, img_save_path)
                
                experiment.log_image(
                    img_save_path,
                    name=f"cfg_{cfg_scale}",
                    step=step
                )
                
                
                selected_indices = list(range(0, len(trajectory), 5))
                images_list = []
                for idx in selected_indices:
                    frame = trajectory[idx]
                    
                    frame = (frame + 1) / 2
                    frame = frame.clamp(0, 1)
                    
                    grid = make_grid(frame, nrow=10)
                    images_list.append(
                        grid.permute(1, 2, 0).cpu().numpy() * 255
                    )
                
                clip = mpy.ImageSequenceClip(images_list, fps=10)
                gif_path = f"/mnt/nvme/images_cifar10/step{step}_cfg{cfg_scale}.gif"
                clip.write_gif(gif_path)
                
                experiment.log_image(
                    gif_path,
                    name=f"trajectory_cfg_{cfg_scale}",
                    step=step,
                    image_format="gif"
                )
    
    losses = []
    sigma_min = 1e-06
    training_cfg_rate = 0.2
    lambda_weight = 0.001
    use_immiscible = True
    gradient_clip = 1.0
    
    remaining_steps = n_steps - start_step
    
    with tqdm(range(remaining_steps), dynamic_ncols=True) as pbar:
        pbar.set_description(f"Training DiT on CIFAR-10 (starting from step {start_step})")
        for idx in pbar:
            step = start_step + idx
            data = next(train_dataloader)
            optimizer.zero_grad()
            
           
            x1 = data[0].to(device)
            x1 = x1 * 2 - 1
            y = data[1].to(device)
            b = x1.shape[0]

           
            t = torch.rand(b, device=device)

            alpha_t = t
            sigma_t = 1 - (1 - sigma_min) * t

           
            alpha_t = alpha_t.view(b, 1, 1, 1)
            sigma_t = sigma_t.view(b, 1, 1, 1)

          
            if use_immiscible:
                k = 4
               
                z_candidates = torch.randn(b, k, image_channels, image_size, image_size, device=x1.device, dtype=x1.dtype)
                
                x1_flat = x1.flatten(start_dim=1)  # [b, c*h*w]
                z_candidates_flat = z_candidates.flatten(start_dim=2)  # [b, k, c*h*w]
                
                
                distances = torch.norm(x1_flat.unsqueeze(1) - z_candidates_flat, dim=2)  # [b, k]
                
                
                min_distances, min_indices = torch.min(distances, dim=1)  # [b]
                
               
                batch_indices = torch.arange(b, device=x1.device)
                z = z_candidates[batch_indices, min_indices]  # [b, c, h, w]
                
            else:
                # Standard noise sampling
                z = torch.randn_like(x1)
                
           
            x_t = sigma_t * z + alpha_t * x1

           
            u_positive = x1 - (1 - sigma_min) * z 
            
         
            if b > 1:
                perm = torch.randperm(b, device=x1.device)
                # Ensure no self-pairing
                for i in range(b):
                    if perm[i] == i:
                        perm[i] = (i + 1) % b
                u_negative = u_positive[perm]
            else:
                u_negative = u_positive

            if training_cfg_rate > 0:
                drop_mask = torch.rand(b, device=x1.device) < training_cfg_rate
                y = torch.where(drop_mask, num_classes, y)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(x_t, t, y)
                
                # Compute losses
                positive_loss = F.mse_loss(pred, u_positive)
                negative_loss = F.mse_loss(pred, u_negative)
                loss = positive_loss - lambda_weight * negative_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            update_ema(ema_model, model, ema_decay)

            # Logging
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})
            experiment.log_metric("loss", loss.item(), step=step)
            experiment.log_metric("grad_norm", grad_norm.item(), step=step)
            experiment.log_metric("positive_loss", positive_loss.item(), step=step)
            experiment.log_metric("negative_loss", negative_loss.item(), step=step)

            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                experiment.log_metric("avg_loss_100", avg_loss, step=step)
                
            if step % 2000 == 0 or step == n_steps - 1:
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"\nStep: {step+1}/{n_steps} | avg_loss: {avg_loss:.4f}")
                losses.clear()
                
                sample_and_log_images()

            if step % 10000 == 0 or step == n_steps - 1:
                # FID evaluation
                try:
                    print(f"Running FID evaluation on {fid_subset_size} samples...")
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        fid_score = fid_eval.fid_score()
                    print(f"FID score at step {step}: {fid_score}")
                    experiment.log_metric("FID", fid_score, step=step)
                except Exception as e:
                    print(f"FID evaluation failed: {e}")
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_root_path, f"step_{step}.pth")
                state_dict = {
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {
                        "dataset": "CIFAR-10",
                        "image_size": image_size,
                        "image_channels": image_channels,
                        "sigma_min": sigma_min,
                        "lambda_weight": lambda_weight,
                        "training_cfg_rate": training_cfg_rate,
                        "gradient_clip": gradient_clip,
                        "ema_decay": ema_decay,
                    }
                }
                torch.save(state_dict, checkpoint_path)
                
                experiment.log_model(
                    name=f"checkpoint_step_{step}",
                    file_or_folder=checkpoint_path
                )

    checkpoint_path = os.path.join(checkpoint_root_path, "model_cifar10_final.pth")
    state_dict = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "config": {
            "dataset": "CIFAR-10",
            "image_size": image_size,
            "image_channels": image_channels,
            "sigma_min": sigma_min,
            "lambda_weight": lambda_weight,
            "training_cfg_rate": training_cfg_rate,
            "gradient_clip": gradient_clip,
            "ema_decay": ema_decay,
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