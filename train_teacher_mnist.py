from dit import DiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit
from copy import deepcopy
from collections import OrderedDict

from model import RectifiedFlow, LogitNormalCosineScheduler
from fid_evaluation import FIDEvaluation

import moviepy.editor as mpy
from comet_ml import Experiment
import os
import torch.optim as optim
import torch.nn.functional as F


def main():
    n_steps = 100000  # Reduced for MNIST (simpler dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128  # Can use larger batch size for MNIST
    
    # MNIST image settings
    image_size = 28  # MNIST native size (or use 32 for power of 2)
    image_channels = 1  # MNIST is grayscale
    num_classes = 10  # MNIST has 10 digit classes (0-9)
    
    # Create directories
    os.makedirs('/mnt/nvme/images_mnist', exist_ok=True)
    os.makedirs('/mnt/nvme/results_mnist', exist_ok=True)
    checkpoint_root_path = '/mnt/nvme/checkpoint/dit_mnist/'
    os.makedirs(checkpoint_root_path, exist_ok=True)

    # Initialize Comet ML experiment
    experiment = Experiment(
        project_name="dit-flow-matching-mnist",
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "dataset": "MNIST",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "model": "DiT-MNIST",
        "dim": 256,  # Smaller model for MNIST
        "depth": 8,   # Fewer layers needed
        "num_heads": 4,  # Fewer attention heads
        "patch_size": 4,  # Larger patches for MNIST (28/4 = 7x7 patches)
        "dropout_prob": 0.1,
        "optimizer": "Adam",
        "mixed_precision": "bfloat16",
        "fid_subset_size": 1000,
        "image_size": image_size,
        "image_channels": image_channels,
        "training_cfg_rate": 0.2,
        "lambda_weight": 0.05,
        "sigma_min": 1e-06,
    })

    # Dataset with normalization to [-1, 1]
    # For MNIST, we need to handle the single channel properly
    transform = T.Compose([
        T.Resize((image_size, image_size)),  # Ensure consistent size
        T.ToTensor(),  # Converts to [0, 1]
        T.Normalize((0.5,), (0.5,)),  # Convert to [-1, 1] for single channel
    ])
    
    dataset = torchvision.datasets.MNIST(
        root="/mnt/nvme/",
        train=True,
        download=True,
        transform=transform,
    )

    # Initialize Logit-Normal + Cosine Scheduler (SD3 approach)
    timestep_scheduler = LogitNormalCosineScheduler(
        loc=0.0,        # SD3 default: symmetric around t=0.5
        scale=1.0,      # SD3 default: moderate focus on intermediate timesteps
        min_t=1e-8,     # Avoid singularities
        max_t=1.0-1e-8  # Avoid singularities
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
        num_workers=8  # Reduced from 40
    )
    train_dataloader = cycle(train_dataloader)

    # Create model for MNIST generation
    model = DiT(
        input_size=image_size,
        patch_size=4,  # Larger patch size for MNIST
        in_channels=image_channels,
        dim=256,  # Smaller dimension
        depth=8,  # Fewer layers
        num_heads=4,  # Fewer heads
        num_classes=num_classes,
        learn_sigma=False,
        class_dropout_prob=0.1,
    ).to(device)
    
    # Create EMA model
    ema_model = deepcopy(model).eval()
    ema_decay = 0.9999
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)
    
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

    # FID evaluation setup for MNIST
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
    
    fid_dataset = torchvision.datasets.MNIST(
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
            
            # Sample from model
            ema_model.eval()
            with torch.no_grad():
                # Use the model's own sampling method
                samples, trajectory = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
                
                # For grayscale images, we need to handle the channel dimension
                if image_channels == 1:
                    # Convert grayscale to RGB for visualization
                    samples_rgb = samples.repeat(1, 3, 1, 1)
                    log_img = make_grid(samples_rgb, nrow=10, normalize=True, value_range=(-1, 1))
                else:
                    log_img = make_grid(samples, nrow=10, normalize=True, value_range=(-1, 1))
                
                img_save_path = f"/mnt/nvme/images_mnist/step{step}_cfg{cfg_scale}.png"
                save_image(log_img, img_save_path)
                
                experiment.log_image(
                    img_save_path,
                    name=f"cfg_{cfg_scale}",
                    step=step
                )
                
                # Create GIF from trajectory
                selected_indices = list(range(0, len(trajectory), 5))
                images_list = []
                for idx in selected_indices:
                    frame = trajectory[idx]
                    # Unnormalize frame for visualization [-1,1] to [0,1]
                    frame = (frame + 1) / 2
                    frame = frame.clamp(0, 1)
                    
                    # Convert grayscale to RGB for GIF
                    if image_channels == 1:
                        frame_rgb = frame.repeat(1, 3, 1, 1)
                        grid = make_grid(frame_rgb, nrow=10)
                    else:
                        grid = make_grid(frame, nrow=10)
                    
                    images_list.append(
                        grid.permute(1, 2, 0).cpu().numpy() * 255
                    )
                
                clip = mpy.ImageSequenceClip(images_list, fps=10)
                gif_path = f"/mnt/nvme/images_mnist/step{step}_cfg{cfg_scale}.gif"
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
    
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training DiT on MNIST")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            
            # Get images and labels
            x1 = data[0].to(device)  # Already in [-1, 1] due to normalization
            y = data[1].to(device)
            b = x1.shape[0]

            # t = timestep_scheduler.sample_timesteps(b, device)
            t = torch.rand(b, device=device)

            alpha_t = t
            sigma_t = 1 - (1 - sigma_min) * t

            # Reshape for broadcasting
            alpha_t = alpha_t.view(b, 1, 1, 1)
            sigma_t = sigma_t.view(b, 1, 1, 1)

            # sample noise using immiscible training
            if use_immiscible:
                k = 4
                # Generate k noise samples for each data point
                z_candidates = torch.randn(b, k, image_channels, image_size, image_size, device=x1.device, dtype=x1.dtype)
                
                x1_flat = x1.flatten(start_dim=1)  # [b, c*h*w]
                z_candidates_flat = z_candidates.flatten(start_dim=2)  # [b, k, c*h*w]
                
                # Compute distances between each data point and its k noise candidates
                distances = torch.norm(x1_flat.unsqueeze(1) - z_candidates_flat, dim=2)  # [b, k]
                
                # Find the farthest noise sample for each data point (for immiscible)
                min_distances, min_indices = torch.min(distances, dim=1)  # [b]
                
                # Method 1: Using gather with proper indexing
                batch_indices = torch.arange(b, device=x1.device)
                z = z_candidates[batch_indices, min_indices]  # [b, c, h, w]
                
            else:
                # Standard noise sampling
                z = torch.randn_like(x1)
                
            # Interpolate between noise and data
            x_t = sigma_t * z + alpha_t * x1

            # Target velocity
            # u_positive = timestep_scheduler.get_velocity_target(x1, z, sigma_min)
            u_positive = x1 - (1 - sigma_min) * z
            # Create negative samples for contrastive loss
            if b > 1:
                perm = torch.randperm(b, device=x1.device)
                # Ensure no self-pairing
                for i in range(b):
                    if perm[i] == i:
                        perm[i] = (i + 1) % b
                u_negative = u_positive[perm]
            else:
                u_negative = u_positive

            # Classifier-free guidance dropout
            if training_cfg_rate > 0:
                drop_mask = torch.rand(b, device=x1.device) < training_cfg_rate
                y = torch.where(drop_mask, num_classes, y)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Forward pass - t should be 1D
                pred = model(x_t, t, y)
                
                # Compute losses
                positive_loss = F.mse_loss(pred, u_positive)
                negative_loss = F.mse_loss(pred, u_negative)
                loss = positive_loss - lambda_weight * negative_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Calculate and clip gradient norm
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
                
            if step % 1000 == 0 or step == n_steps - 1:  # More frequent for MNIST
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"\nStep: {step+1}/{n_steps} | avg_loss: {avg_loss:.4f}")
                losses.clear()
                
                sample_and_log_images()

            if step % 2500 == 0 or step == n_steps - 1:  # More frequent for MNIST
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
                        "dataset": "MNIST",
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

    # Final save
    checkpoint_path = os.path.join(checkpoint_root_path, "model_mnist_final.pth")
    state_dict = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "config": {
            "dataset": "MNIST",
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