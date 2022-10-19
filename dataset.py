import torchvision.transforms as T
import torchvision
import torch


def get_dataset(cfg):
    imsize = cfg.imsize
    batchsize = cfg.batchsize

    transform = T.Compose(
        [
            T.Resize(imsize),
            T.CenterCrop(imsize),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=batchsize, shuffle=True, num_workers=2
    )

    return dataloader
