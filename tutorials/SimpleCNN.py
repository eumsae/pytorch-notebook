from torch import nn
from torch import optim

from SimpleDNN import train, test, get_data_and_loaders
from SimpleDNN import DEVICE


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        
        # 1x28x28 -> 32x14x14
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # 32x14x14 -> 64x6x6
        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # 64x6x6(2304) -> 600
        self.fc_stack_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, 600),  # flatten
            nn.Dropout(0.25))

        # 600 -> 120 -> 10
        self.fc_stack_2 = nn.Sequential(
            nn.Linear(600, 120),
            nn.Linear(120, 10))

    def forward(self, x):
        x = self.conv_stack_1(x)
        x = self.conv_stack_2(x)
        x = self.fc_stack_1(x)
        logits = self.fc_stack_2(x)
        return logits


def __train_test():
    # model
    model = CNN().to(DEVICE)

    # hyper-parameters
    lr = 1e-3  # 0.001
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    batch_size = 64

    # data & data loaders
    data, loaders = get_data_and_loaders(batch_size)
    _, test_data = data
    train_dataloader, test_dataloader = loaders

    # train, test
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} -----")
        train(model, train_dataloader, loss_fn, optimizer)
        test(model, test_dataloader, loss_fn)
    print("Done.")


if __name__ == "__main__":
    __train_test()
