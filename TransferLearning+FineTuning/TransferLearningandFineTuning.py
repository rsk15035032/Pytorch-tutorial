# Import 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm



# Set a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5



# Load pretrained and modified it.
model = torchvision.models.vgg16(weights="DEFAULT")

# If you want to do finetuning then set requires_grad = False
# Remove these two lines if you want to train entire model,
# and only want to load the pretrain weights.

for param in model.parameters():
    param.requires_grad = False

model.avgpool = nn.Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, num_classes))
model.to(device)


#Load data
train_dataset = datasets.CIFAR10(root= 'dataset/',train= True, transform=transforms.ToTensor(),download=True )
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#train networks

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
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

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")

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



