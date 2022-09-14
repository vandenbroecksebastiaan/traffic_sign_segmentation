import torch
import torchvision
from copy import deepcopy


def load_data(batch_size: int):
    # Apply the following transformations
    transforms = [
        torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((50, 50))
        ]),
        # Brightness transformations
        torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((50, 50))
        ])#,
        # Contrast transformations
        # torchvision.transforms.Compose([
        #     torchvision.transforms.ColorJitter(contrast=5),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Resize((50, 50))
        # ]),
        # Hue transformations
        # torchvision.transforms.Compose([
        #     torchvision.transforms.ColorJitter(hue=0.01),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Resize((50, 50))
        # ]),
        # Saturation transformations
        # torchvision.transforms.Compose([
        #     torchvision.transforms.ColorJitter(saturation=5),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Resize((50, 50))
        # ])
    ]

    # Load the dataset and split it in a train and validation part
    dataset = torchvision.datasets.GTSRB(root="data/", download=True,
                                         transform=transforms[0])
    train_set_reference, val_set = torch.utils.data.random_split(
        dataset, [int(len(dataset)*0.9), int(len(dataset)*0.1)]
    )

    # Apply transformations to the train set
    transformed_train_sets = []
    for transform in transforms:
        train_set = deepcopy(train_set_reference)
        train_set.dataset.transforms = transform
        transformed_train_sets.append(train_set)

    train_set = torch.utils.data.ConcatDataset(transformed_train_sets)

    # Create dataloaders from datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=True)

    return train_loader, val_loader