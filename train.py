import torch
import torchvision
from tqdm import tqdm
import os
import json
from functools import partial

from models.dcgan_upsample import get_models
from cfg import CFG
from util import tensor2var, compute_gradient_penalty
from dataset import dataloader
from datetime import datetime
import numpy as np


def manipulate_layers(mode, model):
    modes = {"freeze": False, "unfreeze": True}
    for p in model.parameters():
        p.requires_grad = modes[mode]


def update_stats(statsfile: str, stats: dict):
    with open(statsfile, "w") as f:
        json.dump(stats, f)


FROM_CHECKPOINT = "runs/2022-10-18 17:23:25.821549"

G, D = get_models()
g_optimizer = torch.optim.Adam(G.parameters(), CFG.g_lr, [CFG.beta1, CFG.beta2])
d_optimizer = torch.optim.Adam(D.parameters(), CFG.d_lr, [CFG.beta1, CFG.beta2])
fixed_z = tensor2var(torch.randn(CFG.batchsize, CFG.z_dim))

if FROM_CHECKPOINT:
    G.load_state_dict(
        torch.load(os.path.join(FROM_CHECKPOINT, "generator_checkpoint.pth"))
    )
    D.load_state_dict(
        torch.load(os.path.join(FROM_CHECKPOINT, "discriminator_checkpoint.pth"))
    )


runs_folder = f"runs/{datetime.now()}"
images_folder = os.path.join(runs_folder, "results")
os.makedirs(runs_folder)
os.makedirs(images_folder)
torch.save(fixed_z, os.path.join(runs_folder, "fixed_z.pt"))
run_stats = {"config": CFG().__dict__}
run_stats["losses_G_D"] = []
update_stats = partial(update_stats, f"{runs_folder}/run_stats.json")
update_stats(run_stats)


for epoch in range(CFG.num_epochs):
    g_losses = []
    d_losses = []
    for i, (real_images, _) in enumerate(tqdm(dataloader)):

        # Prepare models
        manipulate_layers("unfreeze", D)
        D.train()
        G.train()
        D.zero_grad()

        real_images = tensor2var(real_images)
        d_out_real = D(real_images)
        d_loss_real = -torch.mean(d_out_real)

        noise = tensor2var(torch.randn(real_images.size(0), CFG.z_dim))
        fake_images = G(noise)
        d_out_fake = D(fake_images)
        d_loss_fake = torch.mean(d_out_fake)

        d_loss = d_loss_real + d_loss_fake
        grad = compute_gradient_penalty(D, real_images, fake_images)
        d_loss = CFG.lambda_gp * grad + d_loss
        d_losses.append(d_loss.item())
        d_loss.backward()
        d_optimizer.step()

        # train the generator every g_num steps
        if i % CFG.g_num == 0:

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

    if not epoch % CFG.save_checkpoint_every:
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
