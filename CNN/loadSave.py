# Import 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms



# Create a fully connected network

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# TO DO: Create a simple CNN.

class CNN(nn.Module):
    def __init__(self, in_channels= 1, num_classes= 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("===> Saving Checkpoints")
    torch.save(state, filename) 



def load_checkpoint(checkpoint_path, model, optimizer):
    print("===> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# Set a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10
load_model = True

#Load data
train_dataset = datasets.MNIST(root= 'dataset/',train= True, transform=transforms.ToTensor(),download=True )
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root= 'dataset/',train= False, transform=transforms.ToTensor(),download=True )
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Initialize network
model = CNN().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

if load_model:
    load_checkpoint("my_checkpoint.pth.tar", model, optimizer)

#train networks

for epoch in range(num_epochs):
    losses = []

    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)


    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets= targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backwards
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or Adam steps
        optimizer.step()

# Check accuracy on train and test sets to see how good our model is.
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on traing datasets")
    else:
        print("checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct = (predictions == y).sum()
            num_samples = predictions.size(0)

        print(f'Got (num_correct)/(num_samples) with an accuracy {(num_correct)/(num_samples)*100:.2f}')
    model.train()
    
    
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)


