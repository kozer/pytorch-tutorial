import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # relu etc
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fully connected network


class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # 28x28 = 784
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Testing if returns correct shape
# model = NN(784, 10)
# x = torch.randn(64, 784)
# res = model(x)
# print(res.shape)  # 64 x 10

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# Hyperparameters
input_size = 784  # 28x28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Load data
train_dataset = datasets.MNIST(root='dataset/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = datasets.MNIST(root='dataset/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train network

for epoch in range(num_epochs):
    for idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # print(data.shape)
        # Get to correct shape
        data = data.reshape(data.shape[0], -1)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model is
def check_acc(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on testing data')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            # 64 x 10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(
            f'Got {num_correct} / {num_samples} with acc {float(num_correct)/float(num_samples) * 100:.2f}'
        )
    model.train()


check_acc(train_loader, model)
check_acc(test_loader, model)
