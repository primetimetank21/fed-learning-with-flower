# pylint: disable-all

# Imports
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List
import numpy as np
import flwr as fl
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

# for NLP
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# use this: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

# pylint disables (temporary)
# pylint: disable=no-member
# pylint: disable=redefined-outer-name)


# for NLP
tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split="train")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(DEVICE), text_list.to(DEVICE), offsets.to(DEVICE)


# Dataset Function
def load_datasets():
    # # Download and transform CIFAR-10 (train and test)
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    # CIFAR10
    # trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    # testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    # # Download and transform MNIST (train and test)
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    # )
    # #MNIST
    # trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    # testset = MNIST("./dataset", train=False, download=True, transform=transform)

    # NLP
    trainset = list(AG_NEWS(split="train"))
    testset = list(AG_NEWS(split="test"))

    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator())

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator())
        trainloaders.append(
            DataLoader(
                ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
            )
        )
        valloaders.append(
            DataLoader(ds_val, batch_size=BATCH_SIZE, collate_fn=collate_batch)
        )
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    return trainloaders, valloaders, testloader


# Constants
DEVICE = torch.device("cpu")
NUM_CLIENTS = 10
BATCH_SIZE = 32
LR = 5  # learning rate

trainloaders, valloaders, testloader = load_datasets()


# Helper functions
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# def train(net, trainloader, epochs: int, verbose=False):
#     """Train the network on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(net.parameters())
#     net.train()
#     for epoch in range(epochs):
#         correct, total, epoch_loss = 0, 0, 0.0
#         for images, labels in trainloader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = net(images)
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()
#             # Metrics
#             epoch_loss += loss
#             total += labels.size(0)
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#         epoch_loss /= len(testloader.dataset)
#         epoch_acc = correct / total
#         if verbose:
#             print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


# def test(net, testloader):
#     """Evaluate the network on the entire test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, total, loss = 0, 0, 0.0
#     net.eval()
#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     loss /= len(testloader.dataset)
#     accuracy = correct / total
#     return loss, accuracy


def train(net, trainloader, epochs: int, verbose=False):
    net.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    # start_time = time.time()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    for idx, (label, text, offsets) in enumerate(trainloader):
        optimizer.zero_grad()
        predicted_label = net(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epochs, idx, len(trainloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0


def test(net, testloader):
    net.eval()
    total_acc, total_count, loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(testloader):
            predicted_label = net(text, offsets)
            loss += criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    loss /= len(testloader.dataset)
    return loss, total_acc / total_count


# Models

# for CIFAR-10
# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# for MNIST
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)

# for NLP
class Net(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(Net, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
