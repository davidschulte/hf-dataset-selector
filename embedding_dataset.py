import numpy as np
from torch.utils.data import Dataset
import os


class EmbeddingDataset(Dataset):

    def __init__(self, filepath):
        self.x = np.loadtxt(os.path.join(filepath, 'standard_embeddings.csv'))
        self.y = np.loadtxt(os.path.join(filepath, 'trained_embeddings.csv'))

        assert len(x) == len(y)

        self.num_rows = len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.num_rows
