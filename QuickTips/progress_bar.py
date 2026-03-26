import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Create a simple toy dataset (1000 RGB images of size 224x224)
# ============================================================

x = torch.randn((1000, 3, 224, 224))             # Fake images
y = torch.randint(low=0, high=10, size=(1000,))  # Class labels (0–9)

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# ============================================================
# Simple CNN Model
# ============================================================

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16 * 224 * 224, 10),
)


# ============================================================
# Loss + Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)


# ============================================================
# Training Loop with tqdm Progress Bar
# ============================================================

NUM_EPOCHS = 5

for epoch in range(NUM_EPOCHS):

    loop = tqdm(loader, leave=True)

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (data, targets) in enumerate(loop):

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predictions = outputs.max(1)
        total_correct += (predictions == targets).sum().item()
        total_samples += targets.size(0)

        total_loss += loss.item()

        # Update tqdm progress bar
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(
            loss=loss.item(),
            acc=100 * total_correct / total_samples
        )

print("Training completed!")