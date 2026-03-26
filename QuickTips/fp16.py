# ============================================================
# Imports
# ============================================================

import torch
import torch.nn as nn                  # Neural network layers (Conv2D, Linear, etc.)
import torch.optim as optim            # Optimizers (Adam, SGD, etc.)
import torch.nn.functional as F        # Activation functions like ReLU
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# ============================================================
# CNN Model Definition
# ============================================================

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()

        # First Convolution Block
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolution Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layer
        # MNIST image size = 28x28
        # After conv + pool twice -> 28 → 14 → 7
        self.fc1 = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # Conv1 + ReLU
        x = self.pool(x)            # Downsampling

        x = F.relu(self.conv2(x))   # Conv2 + ReLU
        x = self.pool(x)            # Downsampling

        x = x.reshape(x.shape[0], -1)   # Flatten
        x = self.fc1(x)                 # Final classification layer

        return x


# ============================================================
# Device Configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

'''
# ============================================================
# Device Setup (CPU / GPU Safe)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable mixed precision ONLY if GPU exists
use_fp16 = torch.cuda.is_available()

if use_fp16:
    scaler = torch.cuda.amp.GradScaler()
'''

# ============================================================
# Hyperparameters
# ============================================================

learning_rate = 3e-4
batch_size = 128
num_epochs = 5
num_classes = 10


# ============================================================
# Load MNIST Dataset
# ============================================================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# ============================================================
# Model, Loss Function, Optimizer
# ============================================================

model = CNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# For Mixed Precision Training (FP16)
scaler = torch.cuda.amp.GradScaler()


# ============================================================
# Training Loop
# ============================================================

for epoch in range(num_epochs):

    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device)
        targets = targets.to(device)

        # ---- Forward Pass (with mixed precision) ----
        with torch.cuda.amp.autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)

        # ---- Backward Pass ----
        optimizer.zero_grad()

        scaler.scale(loss).backward()   # Scale loss to avoid FP16 underflow
        scaler.step(optimizer)          # Update weights
        scaler.update()                 # Update scaling factor


    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.4f}")


# ============================================================
# Accuracy Function
# ============================================================

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    acc = 100 * float(num_correct) / float(num_samples)
    print(f"Accuracy: {acc:.2f}%")

    model.train()


# ============================================================
# Final Evaluation
# ============================================================

print("\nTraining Accuracy:")
check_accuracy(train_loader, model)

print("\nTest Accuracy:")
check_accuracy(test_loader, model)