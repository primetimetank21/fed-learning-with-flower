# Imports
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List
import numpy as np
import flwr as fl
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset, random_split

# from torchvision import transforms

# create own Dataset: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# pylint: disable=no-member,redefined-outer-name


class COBA(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, filename, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(filename, "rb") as f:
            data, targets = np.load(f), list(np.load(f))

        partition_idx = round(len(data) * 0.8)

        self.data = data[:partition_idx] if train is True else data[partition_idx:]
        self.targets = (
            targets[:partition_idx] if train is True else targets[partition_idx:]
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # need to return tuple: (data[i] as tensor, targets[i] as int)
        sample = (torch.tensor(self.data[idx]), int(self.targets[idx]))

        if self.transform:
            sample = self.transform(sample)

        return sample


# Dataset Function
def load_datasets():
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    # need to manually split COBA dataset (80/20)
    trainset = COBA(filename="iobt_128_128.npy", train=True)
    testset = COBA(filename="iobt_128_128.npy", train=False)

    # Split training set into NUM_CLIENTS partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator())

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = (
            len(ds) // NUM_CLIENTS
        )  # 13 % validation set --  changed to 13 because of COBA dataset size
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


# Constants
DEVICE = torch.device("cpu")
NUM_CLIENTS = 13  # 10
BATCH_SIZE = 32
trainloaders, valloaders, testloader = load_datasets()


# Helper functions
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(testloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# pylint: disable=pointless-string-statement
"""
def build_model():
    seed=268
    model = Sequential()
    model.add(Conv2D(filters = 8, kernel_size = (3,3), padding = 'same', activation ='relu', input_shape = (128, 128, 3),
                    use_bias=False, kernel_initializer = initializer(seed)))
    model.add(MaxPool2D(pool_size = (2,2), padding = "same"))
    model.add(Flatten())
    model.add(Dense(4, activation='relu', use_bias=False, kernel_initializer = initializer(seed)))
    model.add(Dense(1, activation='sigmoid', use_bias=False, kernel_initializer = initializer(seed)))
    return model

"""

# Models
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=128 * 128,
                out_channels=8,
                kernel_size=(3, 3),
                padding="same",
                bias=False,
            )
        )
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), padding="same"))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=4, out_features=4, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=4, out_features=1, bias=False))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        # model.add(Conv2D(filters = 8, kernel_size = (3,3), padding = 'same', activation ='relu', input_shape = (128, 128, 3),use_bias=False, kernel_initializer = initializer(seed)))
        # model.add(MaxPool2D(pool_size = (2,2), padding = "same"))
        # model.add(Flatten())
        # model.add(Dense(4, activation='relu', use_bias=False, kernel_initializer = initializer(seed)))
        # model.add(Dense(1, activation='sigmoid', use_bias=False, kernel_initializer = initializer(seed)))

        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config=None):
        return get_parameters(self.net)

    def fit(self, parameters, config=None):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config=None):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
