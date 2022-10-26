import torch

from models import get_model
from util import tensor2var


class DummyConfig:
    imsize = 64
    z_dim = 100
    g_conv_dim = 64
    d_conv_dim = 64
    channels = 3
    device = "cpu"


noise = tensor2var(torch.randn(4, DummyConfig.z_dim), DummyConfig.device)


DummyConfig.model = "dcgan_upsample"
G, D = get_model(DummyConfig)
out_g = G(noise)
print(out_g.shape)
out_d = D(out_g)
