"""
This config is only used on fresh runs.
If a checkpoint is selected, the config from that run will be
loaded instead and the number of epochs will be overwritten
with the provided amount.
"""
from dataclasses import dataclass
import click


@dataclass
class DefaultConfig:
    from_checkpoint: str
    num_epochs: int
    model: str
    batchsize: int
    save_checkpoint_every: int = 25
    imsize: int = 64
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


@click.command()
@click.option("--from-checkpoint", default="")
@click.option("--num-epochs", default=500)
@click.option("--model", default="dcgan_upsample")
@click.option("--batchsize", default=64)
def parse_cfg(**kwargs):

    print(DefaultConfig(**kwargs))


if __name__ == "__main__":
    parse_cfg()
