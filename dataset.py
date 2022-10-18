import torchvision.transforms as transform
import torchvision
import torch

from cfg import CFG

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

dataloader = torch.utils.data.DataLoader(
    data, batch_size=CFG.batchsize, shuffle=True, num_workers=2
)
