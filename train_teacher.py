from dit import DiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit
from copy import deepcopy
from collections import OrderedDict

from model import RectifiedFlow
from fid_evaluation import FIDEvaluation

import moviepy.editor as mpy
from comet_ml import Experiment
import os
import torch.optim as optim
import torch.nn.functional as F


# Clean EMA functions
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
    n_steps = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    ema_decay = 0.9999
    
    # Direct image settings
    image_size = 32  # Smaller image size for direct generation
    image_channels = 3  # RGB channels
    
    # Create directories
    os.makedirs('/mnt/nvme/images', exist_ok=True)
    os.makedirs('/mnt/nvme/results', exist_ok=True)
    checkpoint_root_path = '/mnt/nvme/checkpoint/dit_direct/'
    os.makedirs(checkpoint_root_path, exist_ok=True)

    # Initialize Comet ML experiment
    experiment = Experiment(
        project_name="dit-flow-matching-direct",
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "model": "DiT-Direct",
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "patch_size": 2,
        "dropout_prob": 0.1,
        "optimizer": "Adam",
        "mixed_precision": "bfloat16",
        "ema_decay": ema_decay,
        "fid_subset_size": 1000,
        "image_size": image_size,
        "image_channels": image_channels,
        "training_cfg_rate": 0.2,
        "lambda_weight": 0.05,
        "sigma_min": 1e-06,
    })

    # Dataset with normalization to [0, 1]
    dataset = torchvision.datasets.CIFAR10(
        root="/mnt/nvme/",
        train=True,
        download=True,
        transform=T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),  # This converts to [0, 1]
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

    # Create model for direct image generation
    model = DiT(
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
    
    # Create EMA model
    print("Creating EMA model...")
    model_ema = deepcopy(model).to(device)
    requires_grad(model_ema, False)
    model_ema.eval()
    update_ema(model_ema, model, decay=0)  # Full copy
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create samplers
    sampler = RectifiedFlow(
        model,
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=10,
    )
    
    sampler_ema = RectifiedFlow(
        model_ema,
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=10,
    )
    
    scaler = torch.cuda.amp.GradScaler()

    # FID evaluation setup
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
        transform=T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]),
    )
    
    fid_dataloader = torch.utils.data.DataLoader(
        fid_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False, 
        num_workers=4
    )
    
    fid_dataloader_limited = limited_cycle(fid_dataloader, fid_batches_needed)
    fid_eval = FIDEvaluation(batch_size, fid_dataloader_limited, sampler_ema, num_fid_samples=100)
    
    def sample_and_log_images():
        """Sample images from both regular and EMA models"""
        for cfg_scale in [1.0, 2.5, 5.0]:
            print(f"Sampling images at step {step} with cfg_scale {cfg_scale}...")
            
            # Sample from regular model
            model.eval()
            with torch.no_grad():
                # Use the model's own sampling method
                samples, trajectory = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
                
                log_img = make_grid(samples, nrow=10)
                img_save_path = f"/mnt/nvme/images/step{step}_cfg{cfg_scale}.png"
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
                    # Unnormalize frame for visualization
                    frame = (frame + 1) / 2
                    frame = frame.clamp(0, 1)
                    images_list.append(
                        make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                    )
                
                clip = mpy.ImageSequenceClip(images_list, fps=10)
                gif_path = f"/mnt/nvme/images/step{step}_cfg{cfg_scale}.gif"
                clip.write_gif(gif_path)
                
                experiment.log_image(
                    gif_path,
                    name=f"trajectory_cfg_{cfg_scale}",
                    step=step,
                    image_format="gif"
                )
            model.train()

            # Sample from EMA model
            print(f"Sampling from EMA model at step {step} with cfg_scale {cfg_scale}...")
            with torch.no_grad():
                samples_ema, trajectory_ema = sampler_ema.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
                
                log_img_ema = make_grid(samples_ema, nrow=10)
                img_save_path_ema = f"/mnt/nvme/images/step{step}_cfg{cfg_scale}_ema.png"
                save_image(log_img_ema, img_save_path_ema)
                
                experiment.log_image(
                    img_save_path_ema,
                    name=f"ema_cfg_{cfg_scale}",
                    step=step
                )
    
    losses = []
    sigma_min = 1e-06
    training_cfg_rate = 0.2
    lambda_weight = 0.05
    use_immiscible = True
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training DiT Direct")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            
            # Get images and labels
            x1 = data[0].to(device)  # Already in [0, 1]
            y = data[1].to(device)
            b = x1.shape[0]
            
            # Convert to [-1, 1] for training
            x1 = x1 * 2 - 1
            
            # Sample timesteps with cosine schedule
            t = torch.randn([b, 1, 1, 1], device=device)
            t = 1 - torch.cos(t * 0.5 * torch.pi)

            # sample noise using immiscible training
            # Apply immiscible diffusion with KNN
            if use_immiscible:
                k = 4
                # Generate k noise samples for each data point
                z_candidates = torch.randn(b, k, image_channels, image_size, image_size, device=x1.device, dtype=x1.dtype)
            
                x1_flat = x1.flatten(start_dim=1).to(torch.float16)
                z_candidates_flat = z_candidates.flatten(start_dim=2).to(torch.float16)
                
            
                distances = torch.norm(x1_flat.unsqueeze(1) - z_candidates_flat, dim=2)
            
                min_distances, min_indices = torch.min(distances, dim=1)
                
            
                z = torch.gather(
                    z_candidates, 
                    1, 
                    min_indices.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, image_channels, image_size, image_size)
                )[:, 0, :, :]
                
            else:         
                # sample noise p(x_0)
                z = torch.randn_like(x1)
            # Interpolate between noise and data
            x_t = (1 - (1 - sigma_min) * t) * z + t * x1

            # Target velocity
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
                cfg_mask = torch.rand(b, device=x1.device) > training_cfg_rate
                y = y * cfg_mask

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Forward pass
                t_input = t.squeeze()
                pred = model(x_t, t_input, y)
                
                # Compute losses
                positive_loss = F.mse_loss(pred, u_positive)
                negative_loss = F.mse_loss(pred, u_negative)
                loss = positive_loss - lambda_weight * negative_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            update_ema(model_ema, model, decay=ema_decay)

            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            experiment.log_metric("loss", loss.item(), step=step)

            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                experiment.log_metric("avg_loss_100", avg_loss, step=step)
                
            if step % 2000 == 0 or step == n_steps - 1:
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"\nStep: {step+1}/{n_steps} | avg_loss: {avg_loss:.4f}")
                losses.clear()
                
                sample_and_log_images()

            if step % 5000 == 0 or step == n_steps - 1:
                # FID evaluation
                try:
                    print(f"Running FID evaluation on {fid_subset_size} samples...")
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        fid_score = fid_eval.fid_score()
                    print(f"FID score with EMA at step {step}: {fid_score}")
                    experiment.log_metric("FID", fid_score, step=step)
                except Exception as e:
                    print(f"FID evaluation failed: {e}")
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_root_path, f"step_{step}.pth")
                state_dict = {
                    "model": model.state_dict(),
                    "ema": model_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {
                        "image_size": image_size,
                        "image_channels": image_channels,
                        "ema_decay": ema_decay,
                        "sigma_min": sigma_min,
                        "lambda_weight": lambda_weight,
                        "training_cfg_rate": training_cfg_rate,
                    }
                }
                torch.save(state_dict, checkpoint_path)
                
                experiment.log_model(
                    name=f"checkpoint_step_{step}",
                    file_or_folder=checkpoint_path
                )

    # Final save
    checkpoint_path = os.path.join(checkpoint_root_path, "model_direct_final.pth")
    state_dict = {
        "model": model.state_dict(),
        "ema": model_ema.state_dict(),
        "config": {
            "image_size": image_size,
            "image_channels": image_channels,
            "ema_decay": ema_decay,
            "sigma_min": sigma_min,
            "lambda_weight": lambda_weight,
            "training_cfg_rate": training_cfg_rate,
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