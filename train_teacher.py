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


def main():
    n_steps = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    
    # Create directories
    os.makedirs('images', exist_ok=True)

    # Initialize Comet ML experiment
    experiment = Experiment(
        api_key="YOUR_API_KEY",  # Replace with your Comet ML API key
        project_name="dit-flow-matching",
        workspace="YOUR_WORKSPACE",  # Replace with your workspace
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
    })

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
    optimizer = AdamW8bit(model.parameters(), lr=1e-4, weight_decay=0.0)
    FM = TargetConditionalFlowMatcher(sigma=0.0)
    
    sampler = RectifiedFlow(model)
    scaler = torch.cuda.amp.GradScaler()

    fid_eval = FIDEvaluation(batch_size * 2, train_dataloader, sampler)
    
    def sample_and_log_images():
        for cfg_scale in [1.0, 2.5, 5.0]:
            print(
                f"Sampling images at step {step} with cfg_scale {cfg_scale}..."
            )
            traj = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
            log_img = make_grid(traj[-1], nrow=10)
            img_save_path = f"images/step{step}_cfg{cfg_scale}.png"
            save_image(log_img, img_save_path)
            
            # Log image to Comet ML
            experiment.log_image(
                img_save_path,
                name=f"cfg_{cfg_scale}",
                step=step
            )
            
            # Create and save GIF
            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in traj
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
            log_img = make_grid(traj[-1], nrow=10)
            img_save_path = f"images/step{step}_cfg{cfg_scale}_ema.png"
            save_image(log_img, img_save_path)
            
            # Log EMA image to Comet ML
            experiment.log_image(
                img_save_path,
                name=f"ema_cfg_{cfg_scale}",
                step=step
            )
            
            # Create and save EMA GIF
            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in traj
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
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            x1 = data[0].to(device)
            x1 = x1 * 2 - 1 # normalize to [-1, 1]
            y = data[1].to(device)
            x0 = torch.randn_like(x1)
            t = torch.randn((x1.shape[0],), device=device).sigmoid()

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                vt = model(xt, t, y)
                loss = torch.mean((vt - ut) ** 2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_ema(model)

            if not torch.isnan(loss):
                losses.append(loss.item())
                pbar.set_postfix({"loss": loss.item()})
                experiment.log_metric("loss", loss.item(), step=step)

            
            if step % 10000 == 0 or step == n_steps - 1:
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(
                    f"Step: {step+1}/{n_steps} | loss: {avg_loss:.4f}"
                )
                experiment.log_metric("avg_loss_10k", avg_loss, step=step)
                losses.clear()
                
                model.eval()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    sample_and_log_images()
                model.train()
                

            if step % 50000 == 0 or step == n_steps - 1:
                model.eval()
                model_ema.store(model.parameters())
                model_ema.copy_to(model)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    fid_score = fid_eval.fid_score()
                print(f"FID score with EMA at step {step}: {fid_score}")
                
                model_ema.restore(model.parameters())
                model.train()
                
                experiment.log_metric("FID", fid_score, step=step)
                
                # Save checkpoint
                checkpoint_path = f"checkpoint_step{step}.pth"
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
    state_dict = {
        "model": model.state_dict(),
        "ema": model_ema.state_dict(),
    }
    torch.save(state_dict, "model.pth")
    
    # Log final model
    experiment.log_model(
        name="final_model",
        file_or_folder="model.pth"
    )
    
    # End the experiment
    experiment.end()


if __name__ == "__main__":
    main()