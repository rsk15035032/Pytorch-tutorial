import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    """
    Custom Dataset for Carvana Image Segmentation.

    This dataset class:
    - Loads images and masks from directories
    - Converts them into numpy / tensor format
    - Supports Albumentations-style transforms (image + mask together)
    - Is fully CPU friendly and automatically works with GPU via DataLoader
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[callable] = None,
    ) -> None:
        """
        Args:
            image_dir (str): Path to folder containing input images
            mask_dir (str): Path to folder containing segmentation masks
            transform (callable, optional): Albumentations or custom transform
        """

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Read all image filenames once (faster than calling os.listdir every time)
        self.images = sorted(os.listdir(image_dir))

    def __len__(self) -> int:
        """
        Returns total number of samples in dataset
        """
        return len(self.images)

    def _load_image(self, img_path: str) -> np.ndarray:
        """
        Load an image and convert it to RGB numpy array
        """
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.uint8)
        return image

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """
        Load mask, convert to grayscale and normalize to 0/1
        """
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask, dtype=np.float32)

        # Convert mask values from {0,255} -> {0,1}
        mask[mask == 255.0] = 1.0
        return mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one image-mask pair
        """

        # -------------------------
        # Build file paths
        # -------------------------
        img_name = self.images[index]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir, img_name.replace(".jpg", "_mask.gif")
        )

        # -------------------------
        # Load image and mask
        # -------------------------
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # -------------------------
        # Apply augmentations (if provided)
        # Works perfectly with Albumentations
        # -------------------------
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # -------------------------
        # Ensure mask has channel dimension
        # (important for UNet training)
        # -------------------------
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else np.expand_dims(mask, axis=0)

        return image, mask


# =====================================================
# Example usage (CPU friendly + GPU ready)
# =====================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = CarvanaDataset(
        image_dir="data/images/",
        mask_dir="data/masks/",
        transform=None,
    )

    # DataLoader automatically works on CPU and GPU
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,      # Increase if you have strong CPU
        pin_memory=True,    # Speeds up GPU training
    )

    for images, masks in loader:
        print(images.shape)
        print(masks.shape)
        break
