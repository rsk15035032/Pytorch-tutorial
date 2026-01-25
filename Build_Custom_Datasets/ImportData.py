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
from customData import CatsAndDogsDataset



# Set a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
in_channels = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 32
num_epochs = 10

# Load dataset

dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', 
                             transform= transforms.ToTensor())


# Splite train and test datasets.
dataset_size = len(dataset)
train_size = int(0.6 * dataset_size)
test_size = dataset_size - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])


# Load train and test datasets.
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle= True)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size, shuffle=False)


# Load pretrained and modified it.
model = torchvision.models.googlenet(weights="DEFAULT")


# freeze all layers, change final linear layer with num_classes
for param in model.parameters():
    param.requires_grad = False

# final layer is not frozen
model.fc = nn.Linear(in_features=1024, out_features=num_classes)
model.to(device)

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
def check_accuracy(loader, model, name="dataset"):
    print(f"Checking accuracy on {name}")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum().item()
            num_samples += y.size(0)

    acc = 100 * num_correct / num_samples
    print(f"Accuracy: {acc:.2f}%")
    model.train()


check_accuracy(train_loader, model, "train set")
check_accuracy(test_loader, model, "test set")
