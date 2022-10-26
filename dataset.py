import os
import shutil
from typing import Any, Callable, Optional, Tuple
import glob
import hashlib

import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 100000001


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

    if cfg.dataset == "flowers102":
        data = _get_flowers_dataset(transform)
    elif cfg.dataset == "wojaks":
        data = _get_wojak_dataset(transform)

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=batchsize, shuffle=True, num_workers=2
    )

    return dataloader


def _get_flowers_dataset(transform):
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
    return data


def _get_wojak_dataset(transform):
    return Wojaks(root="data", download=True, transform=transform)


class Wojaks(VisionDataset):
    def __init__(
        self,
        root: str = "data",
        transform: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
    ) -> None:

        self.url = "https://archive.org/download/wojak-collections/Wojak%20MEGA%20Collection.zip"
        self.root = root
        self.filename = "Wojak MEGA Collection.zip"
        self.unpacked_folder = os.path.join(root, "Wojak MEGA Collection")
        self.unpacked_folder_rename = os.path.join(root, "wojaks")
        self.transparent_images = ["1596506322786.png", "1590018617967.png"]

        self.md5hash = "1b548acd7b1da5dd9bfe81db10e82e50"
        super().__init__(root, transform=transform)

        if download:
            self.data = self.download(force_download)
        else:
            self.data = self._get_data()

    def _get_data(self):
        data = []
        for ext in [".jpg", "jpeg", ".png"]:
            files = glob.glob(f"{self.unpacked_folder_rename}/*{ext}")
            data.extend(files)
        return data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        target = []
        imfile = self.data[index]
        img = Image.open(imfile)

        if os.path.basename(imfile) in self.transparent_images:
            img = img.convert("RGBA")

        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self, force_download) -> None:
        if not force_download and os.path.exists(self.unpacked_folder_rename):
            _data = self._get_data()
            md5hash = hashlib.md5(str(sorted(_data)).encode()).hexdigest()
            if md5hash == self.md5hash:
                print(
                    "Data already exists. Skipping download. To download"
                    " anyway, set force_download=True"
                )

                return _data

        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=None
        )

        # cleanup
        shutil.rmtree(os.path.join(self.unpacked_folder, "Transparent Template Wojaks"))
        os.rename(self.unpacked_folder, self.unpacked_folder_rename)
        os.remove(os.path.join(self.root, "Wojak MEGA Collection.zip"))
        shutil.rmtree(os.path.join(self.root, "__MACOSX"))

        return self._get_data()
