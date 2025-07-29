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
import wandb


def main():
    n_steps = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128

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

    logger = wandb.init(project="dit-cfm")
    fid_eval = FIDEvaluation(batch_size * 2, train_dataloader, sampler)
    
    def sample_and_log_images():
        log_imgs = []
        log_gifs = []
        for cfg_scale in [1.0, 2.5, 5.0]:
            print(
                f"Sampling images at step {step} with cfg_scale {cfg_scale}..."
            )
            traj = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
            log_img = make_grid(traj[-1], nrow=10)
            img_save_path = f"images/step{step}_cfg{cfg_scale}.png"
            save_image(log_img, img_save_path)
            log_imgs.append(
                wandb.Image(img_save_path, caption=f"cfg_scale: {cfg_scale}")
            )
            # print(f"Saved images to {img_save_path}")
            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in traj
            ]
            clip = mpy.ImageSequenceClip(images_list, fps=10)
            clip.write_gif(f"images/step{step}_cfg{cfg_scale}.gif")
            log_gifs.append(
                wandb.Video(
                    f"images/step{step}_cfg{cfg_scale}.gif",
                    caption=f"cfg_scale: {cfg_scale}",
                )
            )

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
            # print(f"Saved images to {img_save_path}")
            log_imgs.append(
                wandb.Image(
                    img_save_path, caption=f"EMA with cfg_scale: {cfg_scale}"
                )
            )
            
            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in traj
            ]
            clip = mpy.ImageSequenceClip(images_list, fps=10)
            clip.write_gif(f"images/step{step}_cfg{cfg_scale}_ema.gif")
            log_gifs.append(
                wandb.Video(
                    f"images/step{step}_cfg{cfg_scale}_ema.gif",
                    caption=f"EMA with cfg_scale: {cfg_scale}",
                )
            )
            model_ema.restore(model.parameters())
        logger.log({"Images": log_imgs, "Gifs": log_gifs, "step": step})
    
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
                logger.log({"loss": loss.item(), "step": step})

            
            if step % 10000 == 0 or step == n_steps - 1:
                print(
                    f"Step: {step+1}/{n_steps} | loss: {sum(losses) / len(losses):.4f}"
                )
                losses.clear()
                model.eval()
                with torch.autocast(dtype=torch.bfloat16):
                    sample_and_log_images()
                model.train()
                

            if step % 50000 == 0 or step == n_steps - 1:
                model.eval()
                model_ema.store(model.parameters())
                model_ema.copy_to(model)
                
                with torch.autocast(dtype=torch.bfloat16):
                    fid_score = fid_eval.fid_score()
                print(f"FID score with EMA at step {step}: {fid_score}")
                
                model_ema.restore(model.parameters())
                model.train()
                
                wandb.log({"FID": fid_score, "step": step})

    state_dict = {
        "model": model.state_dict(),
        "ema": model_ema.state_dict(),
    }
    torch.save(state_dict, "model.pth")


if __name__ == "__main__":
    main()
