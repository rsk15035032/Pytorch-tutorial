import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        label = int(self.annotations.iloc[index, 1])

        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)
        y_label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, y_label
       