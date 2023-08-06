from typing import Any
import numpy as np

import ignite.distributed as idist
import torchvision
# import torchvision.transforms as T
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from dataloader.officehome import OfficeHome
from dataloader.visda2017 import VisDA2017
from dataloader.domainnet import DomainNet


def setup_data(config: Any):
    """Download datasets and create dataloaders

    Parameters
    ----------
    config: needs to contain `data_path`, `train_batch_size`, `eval_batch_size`, and `num_workers`
    """
    local_rank = idist.get_local_rank()

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    train_transform =  transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if local_rank > 0:
    # Ensure that only rank 0 download the dataset
        idist.barrier()

    if config.dataset.name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=config.data_path,
            train=True,
            download=False,
            transform=train_transform,
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=config.data_path,
            train=False,
            download=False,
            transform=test_transform,
        )
    else:
        if config.dataset.name == "office-home":
            dataset_cls = OfficeHome
        elif config.dataset.name == "visda2017":
            dataset_cls = VisDA2017
        elif config.dataset.name == "domainnet":
            dataset_cls = DomainNet

        else:
            raise NotImplementedError(f"{config.dataset.name} not implemented.")
        
        
        train_dataset = dataset_cls(
            config.dataset.root, config.dataset.domain, transform=train_transform
        )
        val_dataset = dataset_cls(
            config.dataset.root, config.dataset.domain, transform=test_transform
        )

        assert len(train_dataset) == len(val_dataset)

        # split the dataset with indices
        indices = np.random.permutation(len(train_dataset))
        num_train = int(len(train_dataset) * config.data.train_ratio)
        train_dataset = Subset(train_dataset, indices[:num_train])
        val_dataset = Subset(val_dataset, indices[num_train:])

    if local_rank == 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    dataloader_train = idist.auto_dataloader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    dataloader_eval = idist.auto_dataloader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return dataloader_train, dataloader_eval