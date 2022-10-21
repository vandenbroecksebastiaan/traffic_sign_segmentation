import torch.nn as nn
# NOTE: don't do return x!

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=200, kernel_size=7)
        self.conv2 = nn.Conv2d(in_channels=200, out_channels=200, kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=200, out_channels=350, kernel_size=4)

        self.bn1 = nn.BatchNorm2d(num_features=200)
        self.bn2 = nn.BatchNorm2d(num_features=200)
        self.bn3 = nn.BatchNorm2d(num_features=350)

        self.bn4 = nn.BatchNorm1d(num_features=1024)
        self.bn5 = nn.BatchNorm1d(num_features=512)

        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sm = nn.Softmax(dim=1)
        self.dropout_1d = nn.Dropout(p=0.3)
        self.dropout_2d = nn.Dropout2d(p=0.3)

        self.fc1 = nn.Linear(12600, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 43)

    def forward(self, x):

        # Convolution 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.dropout_2d(x)
        x = self.bn1(x)

        # Convolution 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.dropout_2d(x)
        x = self.bn2(x)

        # Convolution 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout_1d(x)
        x = self.relu(x)
        x = self.bn4(x)

        x = self.fc2(x)
        x = self.dropout_1d(x)
        x = self.relu(x)
        x = self.bn5(x)

        x = self.fc3(x)

        return self.sm(x)
