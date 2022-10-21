import torch
import torchvision
from copy import deepcopy


def make_weights_for_balanced_classes(images, nclasses):
    n_images = len(images)
    count_per_class = [0] * nclasses
    for _, image_class in images:
        count_per_class[image_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, (image, image_class) in enumerate(images):
        weights[idx] = weight_per_class[image_class]
    return torch.DoubleTensor(weights)


def load_data(batch_size: int):
    # Apply the following transformations
    resize_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((50, 50))
        ]
    )

    # Load the train and test datasets
    dataset_train = torchvision.datasets.GTSRB(root="data/", download=True,
                                                split="train",
                                                transform=resize_transform)
    test_set = torchvision.datasets.GTSRB(root="data/", download=True,
                                          split="test",
                                          transform=resize_transform)

    # Split the train dataset into train and validation part
    train_set, val_set = torch.utils.data.random_split(
        dataset_train, [int(len(dataset_train)*0.9), int(len(dataset_train)*0.1)]
    )

    # Create a sampler to oversample minority classes in the train set
    weights = make_weights_for_balanced_classes(train_set,
                                                43)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                                                             len(weights))                     

    # Create dataloaders from datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=False, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*4,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size*4,
                                              shuffle=True)

    return train_loader, val_loader, test_loader