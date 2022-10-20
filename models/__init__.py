def get_model(cfg):
    assert cfg.imsize == 64, "imsize must be 64"
    if cfg.model == "dcgan_upsample":
        from models.dcgan_upsample import Generator
        from models.dcgan import Discriminator
    elif cfg.model == "dcgan":
        from models.dcgan import Generator
        from models.dcgan import Discriminator

    elif cfg.model == "sagan":
        from models.sagan import Generator
        from models.sagan import Discriminator

    G = Generator(
        image_size=cfg.imsize,
        z_dim=cfg.z_dim,
        conv_dim=cfg.g_conv_dim,
        channels=cfg.channels,
    ).to(cfg.device)

    D = Discriminator(
        image_size=cfg.imsize,
        conv_dim=cfg.d_conv_dim,
        channels=cfg.channels,
    ).to(cfg.device)

    return G, D