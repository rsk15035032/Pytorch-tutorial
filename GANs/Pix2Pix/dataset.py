import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import config


class MapDataset(Dataset):
    """
    Pix2Pix Map Dataset

    Expects images where:
    Left half  -> Input image
    Right half -> Target image
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist")

        # Keep only image files
        self.list_files = sorted(
            [
                file for file in os.listdir(self.root_dir)
                if file.endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        if len(self.list_files) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = self.root_dir / img_file

        # Load image and force RGB
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Automatically split image in half (no hardcoding 600)
        h, w, c = image.shape
        split_width = w // 2

        input_image = image[:, :split_width, :]
        target_image = image[:, split_width:, :]

        # Apply shared transforms
        augmentations = config.both_transform(
            image=input_image,
            image0=target_image
        )

        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        # Apply separate transforms
        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


# =========================================================
# Quick Test
# =========================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    dataset = MapDataset(config.TRAIN_DIR)
    loader = DataLoader(dataset, batch_size=4)

    for x, y in loader:
        print("Input shape :", x.shape)
        print("Target shape:", y.shape)

        save_image(x, "input_sample.png")
        save_image(y, "target_sample.png")
        break