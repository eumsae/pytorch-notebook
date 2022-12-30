import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # img size: 28x28
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10))  # classes: 10
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


def train(model, dataloader, loss_fn, optimizer):
    model.train()  # set the model to training mode
    n_samples = len(dataloader.dataset)  # the num of samples
    for batch_i, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        preds = model(X)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()  # init grads to 0
        loss.backward()
        optimizer.step()
        if batch_i % 100 == 0:
            loss, prog = loss.item(), batch_i * len(X)
            print(f"Loss: {loss:>7f}, Progress: {prog:>5d}/{n_samples:>5d}")


def test(model, dataloader, loss_fn):
    model.eval()  # set the model to evaluation mode
    n_samples = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():  # fix grads
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X)
            loss = loss_fn(preds, y)
            test_loss += loss.item()
            correct += (preds.argmax(1) == y).type(torch.float).sum().item()
    acc = correct / n_samples
    avg_loss = test_loss / n_samples
    print(f"[Test] acc: {acc:>7f}, avg_loss: {avg_loss:>7f}")


def __train_test():

    # model
    model = DNN().to(DEVICE)

    # hyper-parameters
    lr = 1e-3  # learning rate, 0.001
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    batch_size = 64

    # data
    training_data = datasets.FashionMNIST(
        root="data",train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor())
    
    # data loaders
    train_dataloader = DataLoader(training_data, batch_size)
    test_dataloader = DataLoader(test_data, batch_size)

    # train, test
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} -----")
        train(model, train_dataloader, loss_fn, optimizer)
        test(model, test_dataloader, loss_fn)
    print("Done.")

    # savaing model
    pth = "dnn.pth"
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

    model = DNN()
    model.load_state_dict(torch.load(pth))
    model.eval()

    X, y = test_data[0]
    with torch.no_grad():
        pred = model(X)
        y_, y = classes[pred[0].argmax(0)], classes[y]
        print(f"Predicted: {y_}, Actual: {y}")


if __name__ == "__main__":
    __train_test()
