import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import config


# ==========================================================
# CUSTOM DATASET
# ==========================================================

class MyImageFolder(Dataset):

    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()

        self.root_dir = root_dir
        self.data = []

        # Folder names (classes)
        self.class_names = os.listdir(root_dir)

        # Collect all image paths
        for index, name in enumerate(self.class_names):

            class_folder = os.path.join(root_dir, name)
            files = os.listdir(class_folder)

            # Save (file_name, class_index)
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_file, label = self.data[index]

        root_and_dir = os.path.join(
            self.root_dir,
            self.class_names[label]
        )

        img_path = os.path.join(root_and_dir, img_file)

        # Read image using OpenCV
        image = cv2.imread(img_path)

        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply joint augmentations
        both_transform = config.both_transforms(image=image)["image"]

        # Generate Low Resolution image
        low_res = config.lowres_transform(image=both_transform)["image"]

        # Generate High Resolution image
        high_res = config.highres_transform(image=both_transform)["image"]

        return low_res, high_res


# ==========================================================
# DATASET TEST FUNCTION
# ==========================================================

def test():

    dataset = MyImageFolder(root_dir="data/")

    # CPU safe DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,              # CPU friendly
        shuffle=True,
        num_workers=0              # Windows + CPU safe
    )

    for low_res, high_res in loader:

        print("Low Resolution Shape :", low_res.shape)
        print("High Resolution Shape:", high_res.shape)

        break


if __name__ == "__main__":
    test()