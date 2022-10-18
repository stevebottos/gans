import torch
import torchvision
import torchvision.transforms as transform
from tqdm import tqdm 

from model import Generator, Discriminator
from cfg import CFG
from util import tensor2var, compute_gradient_penalty


transform = transform.Compose(
    [
        transform.Resize(CFG.imsize),
        transform.CenterCrop(CFG.imsize),
        transform.ToTensor(),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_data = torchvision.datasets.Flowers102(
    root="data", download=True, split="train", transform=transform
)
val_data = torchvision.datasets.Flowers102(
    root="data", download=True, split="val", transform=transform
)
test_data = torchvision.datasets.Flowers102(
    root="data", download=True, split="test", transform=transform
)

data = torch.utils.data.ConcatDataset([train_data, val_data, test_data])
data_loader = torch.utils.data.DataLoader(
    data, batch_size=CFG.batchsize, shuffle=True, num_workers=2
)

G = Generator(
    image_size=CFG.imsize,
    z_dim=CFG.z_dim,
    conv_dim=CFG.g_conv_dim,
    channels=CFG.channels,
).to(CFG.device)

D = Discriminator(
    image_size=CFG.imsize,
    conv_dim=CFG.d_conv_dim,
    channels=CFG.channels,
).to(CFG.device)

g_optimizer = torch.optim.Adam(G.parameters(), CFG.g_lr, [CFG.beta1, CFG.beta2])
d_optimizer = torch.optim.Adam(D.parameters(), CFG.d_lr, [CFG.beta1, CFG.beta2])

for epoch in range(CFG.num_epochs):
    for i, (real_images, _) in enumerate(tqdm(data_loader)):

        for p in D.parameters():
            p.requires_grad = True

        # configure input 
        real_images = tensor2var(real_images)
        
        D.train()
        G.train()

        # ==================== Train D ==================
        # train D more iterations than G
        D.zero_grad()

        # compute loss with real images 
        d_out_real = D(real_images)

        d_loss_real = - torch.mean(d_out_real)

        # noise z for generator
        z = tensor2var(torch.randn(real_images.size(0), CFG.z_dim)) # 64, 100

        fake_images = G(z) # (*, c, 64, 64)
        d_out_fake = D(fake_images) # (*,)

        d_loss_fake = torch.mean(d_out_fake)
        
        # total d loss
        d_loss = d_loss_real + d_loss_fake

        # for the wgan loss function
        grad = compute_gradient_penalty(D, real_images, fake_images)
        d_loss = CFG.lambda_gp * grad + d_loss

        d_loss.backward()
        # update D
        d_optimizer.step()

        # train the generator every 5 steps
        if i % CFG.g_num == 0:

            # =================== Train G and gumbel =====================
            for p in D.parameters():
                p.requires_grad = False  # to avoid computation

            G.zero_grad()
            # create random noise 
            fake_images = G(z)

            # compute loss with fake images 
            g_out_fake = D(fake_images) # batch x n

            g_loss_fake = - torch.mean(g_out_fake)
            
            g_loss_fake.backward()
            # update G
            g_optimizer.step()

        if not i%100:
            torchvision.utils.save_image(fake_images, "results/test.jpg")


