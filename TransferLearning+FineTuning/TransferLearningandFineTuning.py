# Import 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision



# Set a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5



class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self, x):
        return x

# Load pretrained and modified it.
model = torchvision.models.vgg16(pretrained = True)
model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)
model.to(device)

#Load data
train_dataset = datasets.CIFAR10(root= 'dataset/',train= True, transform=transforms.ToTensor(),download=True )
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#train networks

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets= targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

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



