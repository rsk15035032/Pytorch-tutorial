import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.root_dir = root_dir
        self.data = []

        # List class folders
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            class_path = os.path.join(root_dir, name)
            files = os.listdir(class_path)

            for file in files:
                self.data.append((file, index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]

        img_path = os.path.join(self.root_dir, self.class_names[label], img_file)

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply augmentations
        image = config.both_transforms(image=image)["image"]

        # Create HR and LR images
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]

        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="new_data/")

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,   # Best for CPU
        shuffle=True
    )
 # loader = DataLoader(dataset, batch_size=1, num_workers=8) if we use 'GPU'
    for low_res, high_res in loader:
        print("Low Resolution Shape:", low_res.shape)
        print("High Resolution Shape:", high_res.shape)
        break


if __name__ == "__main__":
    test()