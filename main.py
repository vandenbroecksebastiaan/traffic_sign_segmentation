import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from model import Model
from data import load_data

BATCH_SIZE = 256
EPOCHS = 10
LR = 0.001
TRAIN = True

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
    ax.set_xticks([])

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


def get_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of params:", params)


def calc_accuracy(output: torch.Tensor, y: torch.Tensor):
    max_output_indices = output.max(dim=1)[1]
    accuracy = (y == max_output_indices).sum() / y.shape[0]
    return accuracy.item()


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
                    format(val_accuracy, "0.4f")
                )

    print("\n")

    return train_losses, val_losses, val_accuracies


def test(model, test_loader):
    model.eval()

    test_losses = []
    predictions = []
    targets = []

    # Enumerate over the test dataloader to track predictions
    for _, (x_test, y_test) in enumerate(test_loader):
        # Save test loss
        x_test, y_test  = next(iter(test_loader))
        x_test, y_test = x_test.cuda(), y_test.cuda()
        output = model(x_test)

        test_loss = torch.nn.CrossEntropyLoss()(output, y_test).item()
        test_losses.append(test_loss)

        # Save predictions and targets
        output_argmax = torch.argmax(output, dim=1)
        predictions.extend(output_argmax.tolist())
        targets.extend(y_test.tolist())

    # Calculate mean test loss and accuracy
    mean_test_loss = sum(test_losses) / len(test_losses)
    test_accuracy = sum([1 for i, j in zip(predictions, targets) if i == j]) \
                        / len(predictions)

    print("Mean test loss", format(round(mean_test_loss, 4), "0.4f"),
          "\nTest accuracy", test_accuracy)

    # Save classification report
    report = classification_report(targets, predictions)
    with open("output/classification_report.txt", "w") as file:
        file.write(report)
        file.close()



def main():
    torch.manual_seed(0)
    train_loader, val_loader, test_loader = load_data(batch_size=BATCH_SIZE)

    model = Model().cuda()
    get_n_params(model)

    if TRAIN:
        # Train the model
        train_losses, val_losses, val_accuracy = \
            train(model, train_loader, val_loader, LR)
        # Save model weights
        torch.save(model.state_dict(), "output/model.pt")
        # Track training information
        plot_loss(train_losses, val_losses, val_accuracy)

    if not TRAIN:
        # Load model weights
        model.load_state_dict(torch.load("output/model.pt"))

    # Obtain statistics
    test(model, test_loader)


if __name__ == "__main__":
    main()
