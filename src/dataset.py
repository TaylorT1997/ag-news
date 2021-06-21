from torch.utils.data import Dataset

import os
import pandas as pd


class AGNewsDataset(Dataset):
    def __init__(self, root_dir="/home/tom/Projects/ag-news", mode="train"):
        data_dir = os.path.join(root_dir, f"data/{mode}.csv")
        self.samples = pd.read_csv(data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        title = self.samples.iloc[idx]["Title"]
        desc = self.samples.iloc[idx]["Description"]
        label = self.samples.iloc[idx]["Class Index"]
        label = label - 1

        return title, desc, label
