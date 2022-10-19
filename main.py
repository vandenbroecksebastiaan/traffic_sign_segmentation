from sklearn.metrics import accuracy_score
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from model import Model
from data import load_data

# TODO: create a notebook with predictions

BATCH_SIZE = 256
EPOCHS = 1
LR = 0.001
TRAIN = False

def plot_image(dataloader, image_idx):
    image = list(enumerate(dataloader))[image_idx][1][0]
    image = image[0, :]
    image = torch.permute(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()


def plot_loss(train_losses, val_losses, val_accuracy):
    # Create a plot with 2 axis
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(train_losses, c="tab:blue")
    ax.plot(val_losses, c="tab:orange")
    ax.set_ylabel("Cross entropy loss")

    sec_ax = ax.twinx()
    sec_ax.plot(val_accuracy, c="tab:green")
    sec_ax.set_ylabel("Accuracy")

    # Add a legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], label="Train loss", color="tab:blue"),
        Line2D([0], [0], label="Validation loss", color="tab:orange"),
        Line2D([0], [0], label="Validation accuracy", color="tab:green")
    ])
    plt.legend(handles=handles)

    # Save the figure
    plt.savefig("output/losses.png", dpi=300, bbox_inches="tight")


def get_n_params():
    model = Model()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of params:", params)


def calc_accuracy(output: torch.Tensor, y: torch.Tensor):
    max_output_indices = output.max(dim=1)[1]
    accuracy = (y == max_output_indices).sum() / y.shape[0]
    return format(accuracy.item(), "0.4f")


def train(model, train_loader, val_loader, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []

    print("Epoch  | Loss    | Validation loss | Validation accuracy")

    for epoch in range(EPOCHS):

        for idx, (x, y) in enumerate(train_loader):
            print(idx, end="\r")

            model.train()
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # Validation
            if idx % 10 == 0:
                model.eval()
                x_val, y_val = next(iter(val_loader))
                x_val, y_val = x_val.cuda(), y_val.cuda()
                val_output = model(x_val)
                val_loss = criterion(val_output, y_val)

                train_loss = loss.item()
                val_loss = val_loss.item()
                val_accuracy = calc_accuracy(val_output, y_val)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                print(
                    epoch, "     |",
                    format(round(train_loss, 4), "0.4f"), " |",
                    format(round(val_loss, 4), "0.4f"), "         |",
                    val_accuracy
                )

    print("\n")

    return train_losses, val_losses, val_accuracies


def test(model, test_loader):
    model.eval()

    # Obtain test loss and accuracy
    x_test, y_test  = next(iter(test_loader))
    x_test, y_test = x_test.cuda(), y_test.cuda()
    output = model(x_test)

    loss = torch.nn.CrossEntropyLoss()(output, y_test).item()
    loss = format(round(loss, 4), "0.4f")
    accuracy = calc_accuracy(output, y_test)

    print("Test cross entropy loss:", loss, "\t", "Test accuracy", accuracy)

    # Obtain test confusion matrix
    output_argmax = torch.argmax(output, dim=1).tolist()
    y_test_list = y_test.tolist()
    confmat = confusion_matrix(y_test_list, output_argmax)

    plt.figure(figsize=(10, 10))
    plt.imshow(confmat, cmap="Blues")
    plt.xticks(range(0, 43), rotation=90)
    plt.yticks(range(0, 43))

    for idx, i in enumerate(confmat):
        print(idx, i)

    plt.savefig("output/confmat.png", dpi=300, bbox_inches="tight")
    
    exit(0)


def main():
    torch.manual_seed(0)
    train_loader, val_loader, test_loader = load_data(batch_size=BATCH_SIZE)

    model = Model().cuda()

    if TRAIN:
        train_losses, val_losses, val_accuracy = \
            train(model, train_loader, val_loader, LR)

        # Save model weights
        torch.save(model.state_dict(), "output/model.pt")

    if not TRAIN:
        # Load model weights
        model.load_state_dict(torch.load("output/model.pt"))

    # Obtain statistics
    test(model, test_loader)

    if TRAIN:
        plot_loss(train_losses, val_losses, val_accuracy)


if __name__ == "__main__":
    main()
