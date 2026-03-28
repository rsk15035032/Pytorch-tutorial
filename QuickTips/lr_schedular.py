"""
Train a simple Neural Network on MNIST using PyTorch
----------------------------------------------------

Concepts covered:
- Dataset loading using torchvision
- DataLoader and mini-batch training
- Model creation using nn.Sequential
- Training loop (forward + backward pass)
- Learning rate scheduler (ReduceLROnPlateau)
- Accuracy calculation
"""

# =====================================================
# 1. Imports
# =====================================================
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# =====================================================
# 2. Device configuration (GPU if available, otherwise CPU)
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================
# 3. Hyperparameters
# =====================================================
INPUT_SIZE = 28 * 28       # MNIST images are 28x28
HIDDEN_SIZE = 50
NUM_CLASSES = 10           # Digits 0–9
LEARNING_RATE = 0.1        # Intentionally high (to show scheduler effect)
BATCH_SIZE = 128
NUM_EPOCHS = 100


# =====================================================
# 4. Simple Fully Connected Neural Network
# =====================================================
"""
Architecture:
Input (784) → Hidden Layer (50) → ReLU → Output Layer (10)

Why flatten?
Because MNIST images are 2D (28x28), but a Linear layer expects 1D input.
"""
model = nn.Sequential(
    nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
).to(device)


# =====================================================
# 5. Load MNIST Dataset
# =====================================================
"""
transforms.ToTensor() converts images to PyTorch tensors
and scales pixel values from [0,255] → [0,1]
"""
train_dataset = datasets.MNIST(
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =====================================================
# 6. Loss function and optimizer
# =====================================================
criterion = nn.CrossEntropyLoss()

"""
Adam is a very good default optimizer.
Even with a high learning rate, the scheduler will reduce it later.
"""
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =====================================================
# 7. Learning Rate Scheduler
# =====================================================
"""
ReduceLROnPlateau reduces learning rate when the loss stops improving.

factor=0.1 → new_lr = lr * 0.1
patience=10 → wait 10 epochs before reducing
"""
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.1,
    patience=10,
    verbose=True
)


# =====================================================
# 8. Training Loop
# =====================================================
for epoch in range(NUM_EPOCHS):

    epoch_losses = []

    for batch_idx, (images, labels) in enumerate(train_loader):

        # Move data to GPU if available
        images = images.to(device)
        labels = labels.to(device)

        # Flatten images: (N, 1, 28, 28) → (N, 784)
        images = images.reshape(images.shape[0], -1)

        # ----------------------
        # Forward pass
        # ----------------------
        outputs = model(images)
        loss = criterion(outputs, labels)

        epoch_losses.append(loss.item())

        # ----------------------
        # Backward pass
        # ----------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Average loss per epoch
    mean_loss = sum(epoch_losses) / len(epoch_losses)

    # Update scheduler (IMPORTANT: send epoch loss)
    scheduler.step(mean_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Loss: {mean_loss:.4f}")


# =====================================================
# 9. Accuracy Function
# =====================================================
def check_accuracy(loader, model):
    """
    Calculates accuracy on a dataset (train or test)
    """

    model.eval()  # set model to evaluation mode

    num_correct = 0
    num_samples = 0

    with torch.no_grad():  # disable gradient calculation
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            # Flatten again
            images = images.reshape(images.shape[0], -1)

            outputs = model(images)

            # Get predicted class
            _, predictions = outputs.max(1)

            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

    accuracy = 100 * float(num_correct) / float(num_samples)
    print(f"Accuracy: {accuracy:.2f}%")

    model.train()  # switch back to training mode


# =====================================================
# 10. Check Training Accuracy
# =====================================================
check_accuracy(train_loader, model)