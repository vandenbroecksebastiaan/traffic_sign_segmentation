import torch
import torchvision


def load_data(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((50, 50))
    ])
    dataset = torchvision.datasets.GTSRB(root="data/", download=True,
                                         transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [20000, 6640])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=True)
    
    return train_loader, val_loader