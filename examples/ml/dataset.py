from typing import TypedDict
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from datasets import load_from_disk

class SimpleDatasetItem(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor


class SimpleDataset(Dataset):
    def __init__(
        self,
        path: str,
        window_len: int,
    ):
        super().__init__()

        self.dataset = load_from_disk(path)
        self.dataset.set_format(type="torch", columns=self.dataset.column_names)

        self.window_len = window_len


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = item["input_ids"]
        input_ids = input_ids[:self.window_len]
        l = input_ids.shape[0]
        
        labels = item["label_ids"] if "label_ids" in item else input_ids.clone()
        
        labels = labels[:self.window_len]
        
        
        return SimpleDatasetItem(
            input_ids=F.pad(input_ids, (0, self.window_len - l), value=0),
            labels=F.pad(labels, (0, self.window_len - l), value=-100),
        )
        