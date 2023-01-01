import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleDNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # (1, 28, 28) -> (784,)
        self.fc_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10))  # classes: 10

    def forward(self, x):
        x = self.flatten(x)
        logits = self.fc_stack(x)
        return logits


def train(model, dataloader, loss_fn, optimizer):
    model.train()
    n_samples = len(dataloader.dataset)

    for batch_i, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        y_ = model(X)

        loss = loss_fn(y_, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_i % 100 == 0:
            loss, prog = loss.item(), batch_i * len(X)
            print(f"Loss: {loss:>7f}, Progress: {prog:>5d}/{n_samples:>5d}")


def test(model, dataloader, loss_fn):
    model.eval()
    n_samples = len(dataloader.dataset)

    total_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_ = model(X)

            loss = loss_fn(y_, y)
            total_loss += loss.item()
            correct += (y_.argmax(1) == y).type(torch.float).sum().item()

    accuracy = correct / n_samples
    avg_loss = total_loss / n_samples
    print(f"[Test] accuracy: {accuracy:>7f}, avg_loss: {avg_loss:>7f}")


def get_data_and_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.FashionMNIST(
        root="data",train=True, download=True, transform=transform)
    testset = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform)

    trainset_loader = DataLoader(trainset, batch_size)
    testset_loader = DataLoader(testset, batch_size)
    
    return (trainset, testset), (trainset_loader, testset_loader)


def __train_test():

    # model
    model = SimpleDNN().to(DEVICE)

    # hyper-parameters
    lr = 1e-3  # learning rate, 0.001
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    batch_size = 64

    # data & data loaders
    data, loaders = get_data_and_loaders(batch_size)
    _, testset = data
    trainset_loader, testset_loader = loaders


    # train, test
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} -----")
        train(model, trainset_loader, loss_fn, optimizer)
        test(model, testset_loader, loss_fn)
    print("Done.")

    # savaing model
    if not os.path.exists("./pth"):
        os.mkdir("./pth")
    pth = "./pth/simple_dnn.pth"
    torch.save(model.state_dict(), pth)
    print(f"Saved model: {pth}")

    # load model and test it
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",]

    model = SimpleDNN()
    model.load_state_dict(torch.load(pth))
    model.eval()

    X, y = testset[0]
    with torch.no_grad():
        pred = model(X)
        y_, y = classes[pred[0].argmax(0)], classes[y]
        print(f"Predicted: {y_}, Actual: {y}")


if __name__ == "__main__":
    __train_test()
