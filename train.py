import torch
import torchvision
from tqdm import tqdm
import os
import json
from functools import partial
import argparse

import click
from datetime import datetime
import numpy as np

from cfg import DefaultConfig
from dataset import get_dataset
from models import get_model
from util import tensor2var, compute_gradient_penalty


def manipulate_layers(mode, model):
    modes = {"freeze": False, "unfreeze": True}
    for p in model.parameters():
        p.requires_grad = modes[mode]


def train(dataset, G, D, cfg):
    def _update_stats(statsfile: str, stats: dict):
        with open(statsfile, "w") as f:
            json.dump(stats, f)

    g_optimizer = torch.optim.Adam(G.parameters(), cfg.g_lr, [cfg.beta1, cfg.beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), cfg.d_lr, [cfg.beta1, cfg.beta2])
    fixed_z = tensor2var(torch.randn(cfg.batchsize, cfg.z_dim), cfg.device)

    if cfg.from_checkpoint:
        print(cfg.from_checkpoint)
        checkpoint_data = torch.load(
            os.path.join(cfg.from_checkpoint, "checkpoint.pth")
        )
        fixed_z = torch.load(os.path.join(cfg.from_checkpoint, "fixed_z.pt"))
        G.load_state_dict(checkpoint_data["generator_weights"])
        D.load_state_dict(checkpoint_data["discriminator_weights"])
        g_optimizer.load_state_dict(checkpoint_data["generator_optimizer"])
        d_optimizer.load_state_dict(checkpoint_data["discriminator_optimizer"])

    runs_folder = f"runs/{datetime.now()}"
    images_folder = os.path.join(runs_folder, "results")
    os.makedirs(runs_folder)
    os.makedirs(images_folder)
    torch.save(fixed_z, os.path.join(runs_folder, "fixed_z.pt"))
    run_stats = {"config": cfg.__dict__}
    run_stats["losses_G_D"] = []
    update_stats = partial(_update_stats, f"{runs_folder}/run_stats.json")
    update_stats(run_stats)

    for epoch in range(cfg.num_epochs):
        g_losses = []
        d_losses = []
        for i, (real_images, _) in enumerate(tqdm(dataset)):

            # Prepare models
            manipulate_layers("unfreeze", D)
            D.train()
            G.train()
            D.zero_grad()

            real_images = tensor2var(real_images, cfg.device)
            d_out_real = D(real_images)
            d_loss_real = -torch.mean(d_out_real)

            noise = tensor2var(torch.randn(real_images.size(0), cfg.z_dim), cfg.device)
            fake_images = G(noise)
            d_out_fake = D(fake_images)
            d_loss_fake = torch.mean(d_out_fake)

            d_loss = d_loss_real + d_loss_fake
            grad = compute_gradient_penalty(D, real_images, fake_images, cfg.device)
            d_loss = cfg.lambda_gp * grad + d_loss
            d_losses.append(d_loss.item())
            d_loss.backward()
            d_optimizer.step()

            # train the generator every g_num steps
            if i % cfg.g_num == 0:

                manipulate_layers("freeze", D)

                G.zero_grad()
                fake_images = G(noise)
                g_out_fake = D(fake_images)
                g_loss = -torch.mean(g_out_fake)

                g_loss.backward()
                g_losses.append(g_loss.item())
                g_optimizer.step()

        # collecting run stats and saving model if applicable
        g_losses = np.round(np.mean(g_losses), 4)
        d_losses = np.round(np.mean(d_losses), 4)
        run_stats["losses_G_D"].append([g_losses, d_losses])
        update_stats(run_stats)

        if not epoch % cfg.save_checkpoint_every:
            checkpoint = {
                "generator_weights": G.state_dict(),
                "discriminator_weights": D.state_dict(),
                "generator_optimizer": g_optimizer.state_dict(),
                "discriminator_optimizer": d_optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{runs_folder}/checkpoint.pth")

        G.eval()
        with torch.no_grad():
            torchvision.utils.save_image(
                G(fixed_z), f"{images_folder}/{epoch}.jpg", normalize=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-checkpoint", type=str, default="")
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--model", type=str, default="dcgan_upsample")
    parser.add_argument("--batchsize", type=int, default=64)
    args = parser.parse_args()
    cfg = DefaultConfig(**args.__dict__)
    dataset = get_dataset(cfg)
    G, D = get_model(cfg)
    train(dataset, G, D, cfg)