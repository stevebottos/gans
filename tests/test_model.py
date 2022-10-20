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


def test_get_model():
    for model in ["sagan", "dcgan", "dcgan_upsample"]:
        DummyConfig.model = model
        G, D = get_model(DummyConfig)
        outdim = torch.Size([4, 3, 64, 64])
        out_g = G(noise)
        out_d = D(out_g)
        assert out_g.size() == outdim, f"size is {out_g.size()}"
        assert len(out_d) == 4