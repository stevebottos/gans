from dataclasses import dataclass


@dataclass
class CFG:
    num_epochs: int = 500
    save_models_every: int = 25
    imsize: int = 64
    batchsize: int = 64
    z_dim: int = 100
    channels: int = 3
    g_conv_dim: int = 64
    d_conv_dim: int = 64
    device: str = "cuda"
    g_lr: float = 0.0001
    d_lr: float = 0.0004
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_gp: int = 10
    g_num: int = 5
