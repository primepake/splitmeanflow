from dit import DiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel
from torchdyn.core import NeuralODE
from tqdm import tqdm
from ema import LitEma
from bitsandbytes.optim import AdamW8bit

from model import RectifiedFlow
from fid_evaluation import FIDEvaluation

import moviepy.editor as mpy
from comet_ml import Experiment
import os
import torch.optim as optim


def main():
    n_steps = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    
    # Create directories
    os.makedirs('/mnt/nvme/images', exist_ok=True)
    os.makedirs('/mnt/nvme/results', exist_ok=True)  # For FID stats
    checkpoint_root_path = '/mnt/nvme/checkpoint/dit/'
    os.makedirs(checkpoint_root_path, exist_ok=True)

    # Initialize Comet ML experiment
    experiment = Experiment(
        project_name="dit-flow-matching",
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "model": "DiT",
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "patch_size": 2,
        "dropout_prob": 0.1,
        "optimizer": "AdamW8bit",
        "mixed_precision": "bfloat16",
        "ema_decay": 0.9999,
        "fid_subset_size": 1000,  # Add FID subset size to params
    })

    dataset = torchvision.datasets.CIFAR10(
        root="/mnt/nvme/",
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

    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        learn_sigma=False,
        class_dropout_prob=0.1,
    ).to(device)
    model_ema = LitEma(model)
    # optimizer = AdamW8bit(model.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    # FM = TargetConditionalFlowMatcher(sigma=0.0)
    
    sampler = RectifiedFlow(model)
    scaler = torch.cuda.amp.GradScaler()

    # FID evaluation with subset
    fid_subset_size = 1000  # Use only 1000 samples for FID evaluation
    
    # Create a sampler wrapper that generates limited samples
    class LimitedSampler:
        def __init__(self, sampler, max_samples=1000):
            self.sampler = sampler
            self.max_samples = max_samples
            self.samples_generated = 0
            
        def sample(self, *args, **kwargs):
            return self.sampler.sample(*args, **kwargs)
            
        def sample_each_class(self, *args, **kwargs):
            return self.sampler.sample_each_class(*args, **kwargs)
    
    # Create a limited dataloader cycle for FID
    def limited_cycle(iterable, max_batches):
        batch_count = 0
        while batch_count < max_batches:
            for i in iterable:
                yield i
                batch_count += 1
                if batch_count >= max_batches:
                    break
    
    # Calculate how many batches we need for subset
    fid_batches_needed = (fid_subset_size + batch_size - 1) // batch_size
    
    # Create FID dataloader
    fid_dataset = torchvision.datasets.CIFAR10(
        root="/mnt/nvme",
        train=True,
        download=False,
        transform=T.Compose([T.ToTensor()]),  # No augmentation for FID
    )
    
    fid_dataloader = torch.utils.data.DataLoader(
        fid_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle to get random subset
        drop_last=False, 
        num_workers=4
    )
    
    # Create limited dataloader
    fid_dataloader_limited = limited_cycle(fid_dataloader, fid_batches_needed)
    
    # Initialize FID evaluation with original signature
    fid_eval = FIDEvaluation(batch_size, fid_dataloader_limited, sampler, num_fid_samples=100)
    
    def sample_and_log_images():
        for cfg_scale in [1.0, 2.5, 5.0]:
            print(
                f"Sampling images at step {step} with cfg_scale {cfg_scale}..."
            )
            traj = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
            # Unpack the tuple: (final_samples, all_steps)
            final_samples, all_steps = traj
            log_img = make_grid(final_samples, nrow=10)
            img_save_path = f"images/step{step}_cfg{cfg_scale}.png"
            save_image(log_img, img_save_path)
            
            # Log image to Comet ML
            experiment.log_image(
                img_save_path,
                name=f"cfg_{cfg_scale}",
                step=step
            )
            
            # Create and save GIF from all steps
            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in all_steps
            ]
            clip = mpy.ImageSequenceClip(images_list, fps=10)
            gif_path = f"images/step{step}_cfg{cfg_scale}.gif"
            clip.write_gif(gif_path)
            
            # Log GIF to Comet ML
            experiment.log_image(
                gif_path,
                name=f"trajectory_cfg_{cfg_scale}",
                step=step,
                image_format="gif"
            )

            # Sample with EMA
            print("Copying EMA to model...")
            model_ema.store(model.parameters())
            model_ema.copy_to(model)
            print(
                f"Sampling images with ema model at step {step} with cfg_scale {cfg_scale}..."
            )
            traj = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
            # Unpack the tuple: (final_samples, all_steps)
            final_samples, all_steps = traj
            log_img = make_grid(final_samples, nrow=10)
            img_save_path = f"images/step{step}_cfg{cfg_scale}_ema.png"
            save_image(log_img, img_save_path)
            
            # Log EMA image to Comet ML
            experiment.log_image(
                img_save_path,
                name=f"ema_cfg_{cfg_scale}",
                step=step
            )
            
            # Create and save EMA GIF from all steps
            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in all_steps
            ]
            clip = mpy.ImageSequenceClip(images_list, fps=10)
            gif_path_ema = f"images/step{step}_cfg{cfg_scale}_ema.gif"
            clip.write_gif(gif_path_ema)
            
            # Log EMA GIF to Comet ML
            experiment.log_image(
                gif_path_ema,
                name=f"ema_trajectory_cfg_{cfg_scale}",
                step=step,
                image_format="gif"
            )
            
            model_ema.restore(model.parameters())
    
    losses = []
    sigma_min = 1e-06
    training_cfg_rate = 0.2
    lambda_weight = 0.05
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            x1 = data[0].to(device)
            x1 = x1 * 2 - 1 # normalize to [-1, 1]

            y = data[1].to(device)
            b = x1.shape[0]

            print(x1.shape, y.shape)

            t = torch.randn([b, 1, 1, 1], device=device)
            t = 1 - torch.cos(t * 0.5 * torch.pi)

            z = torch.randn_like(x1)

            x_t = (1 - (1 - sigma_min) * t) * z + t * x1

            u_positive = x1 - (1 - sigma_min) * z

            if b > 1:
                perm = torch.randperm(b, device=x1.device)
                # Ensure no self-pairing
                for i in range(b):
                    if perm[i] == i:
                        # Swap with next element (circularly)
                        perm[i] = (i + 1) % b

                # Get negative samples
                u_negative = u_positive[perm]
            
            else:
                u_negative = u_positive

            # during training, we randomly drop condition to trade off mode coverage and sample fidelity
            if training_cfg_rate > 0:
                cfg_mask = torch.rand(b, device=x1.device) > training_cfg_rate
                y = y * cfg_mask

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                print(x_t.shape, t.shape, y.shape)
                pred = model(x_t, t, y)
                positive_loss = F.mse_loss(pred, u_positive)
                negative_loss = F.mse_loss(pred, u_negative)

                loss = positive_loss - lambda_weight * negative_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_ema(model)

        
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            experiment.log_metric("loss", loss.item(), step=step)

            avg_loss = sum(losses) / len(losses) if losses else 0
            print(
                f"Step: {step+1}/{n_steps} | loss: {avg_loss:.4f}"
            )
            experiment.log_metric("avg_loss_10k", avg_loss, step=step)
                
            if step % 2000 == 0 or step == n_steps - 1:
                losses.clear()
                
                model.eval()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    sample_and_log_images()
                model.train()
                

            if step % 2000 == 0 or step == n_steps - 1:
                model.eval()
                model_ema.store(model.parameters())
                model_ema.copy_to(model)
                
                try:
                    # Note: FID evaluation might need float32, not bfloat16
                    print(f"Running FID evaluation on {fid_subset_size} samples...")
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        fid_score = fid_eval.fid_score()
                    print(f"FID score with EMA at step {step}: {fid_score}")
                    experiment.log_metric("FID", fid_score, step=step)
                except Exception as e:
                    print(f"FID evaluation failed: {e}")
                    print("Skipping FID evaluation for this step")
                
                model_ema.restore(model.parameters())
                model.train()
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_root_path, f"step_{step}.pth")
                state_dict = {
                    "model": model.state_dict(),
                    "ema": model_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                torch.save(state_dict, checkpoint_path)
                
                # Log model checkpoint to Comet ML
                experiment.log_model(
                    name=f"checkpoint_step_{step}",
                    file_or_folder=checkpoint_path
                )

    # Final save
    checkpoint_path = os.path.join(checkpoint_root_path, "model.pth")
    state_dict = {
        "model": model.state_dict(),
        "ema": model_ema.state_dict(),
    }
    torch.save(state_dict, checkpoint_path)
    
    # Log final model
    experiment.log_model(
        name="final_model",
        file_or_folder=checkpoint_path
    )
    
    # End the experiment
    experiment.end()


if __name__ == "__main__":
    main()