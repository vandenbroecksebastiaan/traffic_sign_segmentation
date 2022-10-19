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

    # Load the train and test datasets
    dataset_train = torchvision.datasets.GTSRB(root="data/", download=True,
                                               split="train",
                                               transform=transforms[0])
    print("dataset_train", len(dataset_train))
    test_set = torchvision.datasets.GTSRB(root="data/", download=True,
                                          split="test",
                                          transform=transforms[0])

    # Split the train dataset into train and validation part
    train_set, val_set = torch.utils.data.random_split(
        dataset_train, [int(len(dataset_train)*0.9), int(len(dataset_train)*0.1)]
    )


    """
    # Apply transformations to the train set
    transformed_train_sets = []
    for transform in transforms:
        train_set = deepcopy(train_set_reference)
        train_set.dataset.transforms = transform
        transformed_train_sets.append(train_set)

    train_set = torch.utils.data.ConcatDataset(transformed_train_sets)
    """

    # Create dataloaders from datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1024,
                                              shuffle=True)

    return train_loader, val_loader, test_loader