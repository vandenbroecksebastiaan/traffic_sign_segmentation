import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from model import Model
from data import load_data


BATCH_SIZE = 512
EPOCHS = 10


def plot_image(dataloader, image_idx):
    image = list(enumerate(dataloader))[image_idx][1][0]
    image = image[0, :]
    image = torch.permute(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()


def plot_loss(train_losses, val_losses, val_accuracy):
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(train_losses, c="tab:blue", label="Train loss")
    ax.plot(val_losses, c="tab:orange", label="Validation loss")
    ax.set_ylabel("Cross entropy loss")
    plt.legend()

    sec_ax = ax.twinx()
    sec_ax.plot(val_accuracy, c="tab:green", label="Validation accuracy")
    sec_ax.set_ylabel("Accuracy")

    plt.legend()
    plt.savefig("output/losses.png", dpi=300, bbox_inches="tight")


def get_n_params():
    model = Model()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of params:", params)


def calc_accuracy(y: torch.Tensor, output: torch.Tensor):
    max_output_indices = output.max(dim=1)[1]
    accuracy = (y == max_output_indices).sum() / y.shape[0]
    return format(accuracy.item(), "0.4f")


def train(model, train_loader, val_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accuracy = []

    for epoch in range(EPOCHS):

        for idx, (x, y) in enumerate(train_loader):

            model.train()
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # Validation
            if idx % int(len(train_loader)) / 3 == 0:
                model.eval()
                x_val, y_val = next(iter(val_loader))
                x_val, y_val = x_val.cuda(), y_val.cuda()
                val_output = model(x_val)
                val_loss = criterion(val_output, y_val)

                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                val_accuracy.append(calc_accuracy(y, output))

                print(
                    epoch, "\t",
                    format(round(loss.item(), 4), "0.4f"),
                    format(round(val_loss.item(), 4), "0.4f"), "\t",
                    calc_accuracy(y, output), "\t",
                    round(optimizer.param_groups[0]['lr'], 4)
                )


    return train_losses, val_losses, val_accuracy


def main():
    torch.manual_seed(0)
    train_loader, val_loader = load_data(batch_size=BATCH_SIZE)

    get_n_params()

    model = Model().cuda()
    train_losses, val_losses, val_accuracy = \
        train(model, train_loader, val_loader)

    plot_loss(train_losses, val_losses, val_accuracy)


if __name__ == "__main__":
    main()