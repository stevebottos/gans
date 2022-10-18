import torch
import torchvision
import torchvision.transforms as transform
from tqdm import tqdm

from model import get_models
from cfg import CFG
from util import tensor2var, compute_gradient_penalty
from dataset import dataloader


def manipulate_layers(mode, model):
    modes = {"freeze": False, "unfreeze": True}
    for p in model.parameters():
        p.requires_grad = modes[mode]


G, D = get_models()
g_optimizer = torch.optim.Adam(G.parameters(), CFG.g_lr, [CFG.beta1, CFG.beta2])
d_optimizer = torch.optim.Adam(D.parameters(), CFG.d_lr, [CFG.beta1, CFG.beta2])
fixed_z = tensor2var(torch.randn(CFG.batchsize, CFG.z_dim))

for epoch in range(CFG.num_epochs):
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

        d_loss.backward()
        d_optimizer.step()

        # train the generator every g_num steps
        if i % CFG.g_num == 0:

            manipulate_layers("freeze", D)

            G.zero_grad()
            fake_images = G(noise)
            g_out_fake = D(fake_images)
            g_loss_fake = -torch.mean(g_out_fake)

            g_loss_fake.backward()
            g_optimizer.step()

    G.eval()
    with torch.no_grad():
        torchvision.utils.save_image(G(fixed_z), f"results/{epoch}.jpg")
