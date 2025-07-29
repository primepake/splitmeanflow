from dit import DiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel
from torchdyn.core import NeuralODE
from tqdm import tqdm
# Remove LitEma import
# from ema import LitEma
from bitsandbytes.optim import AdamW8bit
from diffusers.models import AutoencoderKL

from model import RectifiedFlowVAE as RectifiedFlow
from fid_evaluation import FIDEvaluation

import moviepy.editor as mpy
from comet_ml import Experiment
import os
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict


# Add clean EMA functions
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def main():
    n_steps = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    vae_model = "ema"  # or "mse"
    ema_decay = 0.9999
    
    # Create directories
    os.makedirs('/mnt/nvme/images', exist_ok=True)
    os.makedirs('/mnt/nvme/results', exist_ok=True)  # For FID stats
    checkpoint_root_path = '/mnt/nvme/checkpoint/dit_vae/'
    os.makedirs(checkpoint_root_path, exist_ok=True)

    # Load VAE model
    print("Loading VAE model...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_model}").to(device)
    vae.eval()  # VAE is always in eval mode
    vae.requires_grad_(False)  # Freeze VAE
    
    # VAE latent dimensions
    image_size = 256  # We'll resize CIFAR to this
    latent_size = image_size // 8  # VAE downscales by 8x (32x32 for 256x256 images)
    latent_channels = 4  # VAE latent channels

    # Initialize Comet ML experiment
    experiment = Experiment(
        project_name="dit-flow-matching-vae",
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "model": "DiT-VAE",
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "patch_size": 2,
        "dropout_prob": 0.1,
        "optimizer": "Adam",
        "mixed_precision": "bfloat16",
        "ema_decay": ema_decay,
        "fid_subset_size": 1000,
        "vae_model": vae_model,
        "image_size": image_size,
        "latent_size": latent_size,
        "latent_channels": latent_channels,
    })

    # Dataset with proper normalization for VAE
    dataset = torchvision.datasets.CIFAR10(
        root="/mnt/nvme/",
        train=True,
        download=True,
        transform=T.Compose([
            T.Resize((image_size, image_size)),  # Resize to VAE input size
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
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

    # Create model for latent space
    model = DiT(
        input_size=latent_size,
        patch_size=2,
        in_channels=latent_channels,  # 4 channels for VAE latents
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        learn_sigma=False,
        class_dropout_prob=0.1,
    ).to(device)
    
    # Create EMA model using deepcopy (clean approach)
    print("Creating EMA model...")
    model_ema = deepcopy(model).to(device)  # Create an EMA of the model
    requires_grad(model_ema, False)  # EMA doesn't need gradients
    model_ema.eval()  # EMA model should always be in eval mode
    
    # Initialize EMA with same weights
    update_ema(model_ema, model, decay=0)  # decay=0 means full copy
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create samplers for both models
    sampler = RectifiedFlow(
        model,
        device=device,
        channels=latent_channels,
        image_size=latent_size,
        num_classes=10,
    )
    
    sampler_ema = RectifiedFlow(
        model_ema,
        device=device,
        channels=latent_channels,
        image_size=latent_size,
        num_classes=10,
    )
    
    scaler = torch.cuda.amp.GradScaler()

    # FID evaluation setup (similar to your code)
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
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
    
    # Create a wrapper sampler that decodes latents before FID evaluation
    class DecodingSampler:
        def __init__(self, latent_sampler, vae):
            self.latent_sampler = latent_sampler
            self.vae = vae
            
        def decode_latents(self, latents):
            with torch.no_grad():
                # Denormalize latents
                latents = latents / 0.18215
                images = self.vae.decode(latents).sample
                # Convert from [-1, 1] to [0, 1]
                images = (images + 1) / 2
                return images.clamp(0, 1)
        
        def sample(self, *args, **kwargs):
            latents = self.latent_sampler.sample(*args, **kwargs)
            return self.decode_latents(latents)
            
        def sample_each_class(self, *args, **kwargs):
            result = self.latent_sampler.sample_each_class(*args, **kwargs)
            if isinstance(result, tuple):
                # Handle return_all_steps=True case
                latents, trajectory = result
                decoded = self.decode_latents(latents)
                # Only decode final frame for trajectory to save memory
                return decoded, trajectory
            else:
                return self.decode_latents(result)
    
    # Use EMA sampler for FID evaluation
    decoding_sampler_ema = DecodingSampler(sampler_ema, vae)
    fid_eval = FIDEvaluation(batch_size, fid_dataloader_limited, decoding_sampler_ema, num_fid_samples=100)
    
    def decode_latents(latents):
        """Decode latents back to images using VAE"""
        with torch.no_grad():
            # Denormalize latents
            latents = latents / 0.18215
            images = vae.decode(latents).sample
            # Convert from [-1, 1] to [0, 1]
            images = (images + 1) / 2
            return images.clamp(0, 1)
    
    def sample_and_log_images():
        """Sample images from both regular and EMA models"""
        for cfg_scale in [1.0, 2.5, 5.0]:
            print(f"Sampling images at step {step} with cfg_scale {cfg_scale}...")
            
            # Sample from regular model
            model.eval()
            with torch.no_grad():
                traj = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
                latent_samples, latent_trajectory = traj
                
                # Decode latents to images
                final_images = decode_latents(latent_samples)
                
                log_img = make_grid(final_images, nrow=10)
                img_save_path = f"/mnt/nvme/images/step{step}_cfg{cfg_scale}.png"
                save_image(log_img, img_save_path)
                
                experiment.log_image(
                    img_save_path,
                    name=f"cfg_{cfg_scale}",
                    step=step
                )
                
                # Create GIF (decode every 5th frame to save memory)
                selected_indices = list(range(0, len(latent_trajectory), 5))
                images_list = []
                for idx in selected_indices:
                    decoded_frame = decode_latents(latent_trajectory[idx])
                    images_list.append(
                        make_grid(decoded_frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
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

            # Sample from EMA model (it's already separate, no store/restore needed!)
            print(f"Sampling from EMA model at step {step} with cfg_scale {cfg_scale}...")
            with torch.no_grad():
                traj_ema = sampler_ema.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
                latent_samples_ema, latent_trajectory_ema = traj_ema
                
                final_images_ema = decode_latents(latent_samples_ema)
                
                log_img_ema = make_grid(final_images_ema, nrow=10)
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
    
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training DiT in VAE Latent Space")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            
            # Get images and labels
            images = data[0].to(device)  # Already normalized to [-1, 1]
            y = data[1].to(device)
            b = images.shape[0]
            
            # Encode images to latent space
            with torch.no_grad():
                # VAE encoding
                latent_dist = vae.encode(images).latent_dist
                x1 = latent_dist.sample()
                # Scale latents (important!)
                x1 = x1 * 0.18215

            # Your training logic adapted for latents
            t = torch.randn([b, 1, 1, 1], device=device)
            t = 1 - torch.cos(t * 0.5 * torch.pi)

            z = torch.randn_like(x1)  # Noise in latent space

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
                # Reshape t for the model
                t_input = t.squeeze()  # Remove extra dimensions
                pred = model(x_t, t_input, y)
                positive_loss = F.mse_loss(pred, u_positive)
                negative_loss = F.mse_loss(pred, u_negative)

                loss = positive_loss - lambda_weight * negative_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA model (clean approach)
            update_ema(model_ema, model, decay=ema_decay)

            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            experiment.log_metric("loss", loss.item(), step=step)

            if step % 100 == 0:  # Log average loss more frequently
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                experiment.log_metric("avg_loss_100", avg_loss, step=step)
                
            if step % 2000 == 0 or step == n_steps - 1:
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"\nStep: {step+1}/{n_steps} | avg_loss: {avg_loss:.4f}")
                losses.clear()
                
                sample_and_log_images()

            if step % 5000 == 0 or step == n_steps - 1:
                # FID evaluation using EMA model
                try:
                    print(f"Running FID evaluation on {fid_subset_size} samples...")
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        fid_score = fid_eval.fid_score()
                    print(f"FID score with EMA at step {step}: {fid_score}")
                    experiment.log_metric("FID", fid_score, step=step)
                except Exception as e:
                    print(f"FID evaluation failed: {e}")
                    print("Skipping FID evaluation for this step")
                
                # Save checkpoint with clean state dicts
                checkpoint_path = os.path.join(checkpoint_root_path, f"step_{step}.pth")
                state_dict = {
                    "model": model.state_dict(),
                    "ema": model_ema.state_dict(),  # Now has proper keys!
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {
                        "vae_model": vae_model,
                        "image_size": image_size,
                        "latent_size": latent_size,
                        "latent_channels": latent_channels,
                        "ema_decay": ema_decay,
                    }
                }
                torch.save(state_dict, checkpoint_path)
                
                experiment.log_model(
                    name=f"checkpoint_step_{step}",
                    file_or_folder=checkpoint_path
                )

    # Final save
    checkpoint_path = os.path.join(checkpoint_root_path, "model_vae_final.pth")
    state_dict = {
        "model": model.state_dict(),
        "ema": model_ema.state_dict(),  # Clean state dict!
        "config": {
            "vae_model": vae_model,
            "image_size": image_size,
            "latent_size": latent_size,
            "latent_channels": latent_channels,
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