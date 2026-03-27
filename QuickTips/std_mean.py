"""
Compute the mean and standard deviation of an image dataset (CIFAR-10).

Why this is useful:
-------------------
Most deep learning models train faster and perform better when the input
data is normalized. To normalize correctly, we first need the dataset mean
and standard deviation.

Author: Ravi 
"""

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


# -------------------------------------------------
# 1. Device configuration
# -------------------------------------------------
# If a GPU is available, use it. Otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# 2. Function to compute mean and standard deviation
# -------------------------------------------------
def compute_mean_std(data_loader):
    """
    Computes the mean and standard deviation of a dataset.

    Formula used:
        Var(X) = E[X^2] - (E[X])^2

    Returns:
        mean (Tensor): Mean value for each channel
        std  (Tensor): Standard deviation for each channel
    """

    # Initialize variables
    channels_sum = torch.zeros(3)       # For RGB channels
    channels_squared_sum = torch.zeros(3)
    num_batches = 0

    # Loop over all batches
    for images, _ in tqdm(data_loader, desc="Computing mean & std"):

        # Move images to device if needed
        images = images.to(device)

        # Mean of the batch (per channel)
        channels_sum += torch.mean(images, dim=[0, 2, 3])

        # Mean of squared values (per channel)
        channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])

        num_batches += 1

    # Final mean
    mean = channels_sum / num_batches

    # Final standard deviation
    std = torch.sqrt(channels_squared_sum / num_batches - mean ** 2)

    return mean, std



if __name__ == "__main__":


# -------------------------------------------------
# 3. Load CIFAR-10 dataset
# -------------------------------------------------
# We only apply ToTensor() because we want raw pixel values
# in the range [0, 1] to calculate mean and std correctly.

    dataset = datasets.CIFAR10(
        root="dataset/",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

# -------------------------------------------------
# 4. Create DataLoader
# -------------------------------------------------
# shuffle=True does NOT affect mean/std computation,
# but it is good practice when loading training data.

    loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,     # works now on Windows
        pin_memory=True
    )

    # Compute statistics
    mean, std = compute_mean_std(loader)

    # Print results
    print("\nDataset Statistics (CIFAR-10)")
    print("--------------------------------")
    print(f"Mean: {mean}")
    print(f"Std : {std}")