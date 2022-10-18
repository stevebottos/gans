class CFG:
    num_epochs = 20
    imsize = 64
    batchsize = 8
    z_dim = 100
    channels = 3
    g_conv_dim = 64
    d_conv_dim = 64
    device = "cpu"
    g_lr = 0.0001
    d_lr = 0.0004
    beta1 = 0.5 
    beta2 = 0.999
    lambda_gp = 10 
    g_num = 5
