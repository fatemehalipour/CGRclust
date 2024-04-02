from torch.utils.data import Dataset
from typing import Tuple
import torchvision
import torch


class PairSeqData(Dataset):
    def __init__(self, train_pairs, transform=None):
        self.pairs = train_pairs
        self.transform = transform

    def __len__(self) -> int:
        """
        Return total number of samples
        """
        return self.pairs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one sample of data, original seq and its mimic (X, X')
        """
        original_chaos = torch.Tensor(self.pairs[index][0]).unsqueeze(dim=0)
        mimic_chaos = torch.Tensor(self.pairs[index][1]).unsqueeze(dim=0)
        # original_chaos = torch.div(original_chaos, torch.max(original_chaos))
        # mimic_chaos = torch.div(mimic_chaos, torch.max(mimic_chaos))
        # normalized_original_chaos = torchvision.transforms.functional.normalize(original_chaos, mean=[0.5000], std=[.1000])
        # normalized_mimic_chaos = torchvision.transforms.functional.normalize(original_chaos, mean=[0.5000], std=[.1000])

        if self.transform:
            return self.transform(original_chaos.repeat(3, 1, 1)), self.transform(mimic_chaos.repeat(3, 1, 1))
        else:
            return original_chaos, mimic_chaos


class SeqData(Dataset):
    def __init__(self, sequences, labels, classes, class_to_idx, transform=None):
        self.sequences = sequences
        self.labels = labels
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Returns one sample of data, fcgr and label (X, y)"""
        chaos = torch.Tensor(self.sequences[index]).unsqueeze(dim=0)
        # chaos = torch.div(chaos, torch.max(chaos))
        # normalized_chaos = torchvision.transforms.functional.normalize(chaos, mean=[0.5000], std=[.1000])
        class_idx = self.labels[index]

        # Transform if necessary
        if self.transform:
            return self.transform(chaos.repeat(3, 1, 1)), class_idx  # return data, label (X, y)
        else:
            return chaos, class_idx  # return untranformed data and label
