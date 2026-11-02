import os
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import Dataset

# Use the SAME import that dataloader_tuh.py uses for load_positions
# (adjust this line if the helper lives elsewhere in your repo).
from downstream_tasks.dataloader_tuh import load_positions


class TUSZ(Dataset):
    """
    Reads the output of preprocess_tusz.py:
        <path>/<processed_dir>/<mode>/<recording>_X.npy   (n_windows, C, T) float32, volts
        <path>/<processed_dir>/<mode>/<recording>_y.npy   (n_windows,)      int64, 0-indexed

    Every window of every recording is one sample. Windows are read lazily with
    np.load(mmap_mode="r"), so the whole dataset never has to fit in RAM.
    Splits come from the official TUSZ folders: train -> train, dev -> val, eval -> test.
    """

    def __init__(
        self,
        path: str,
        mode: str,
        positions=None,
        electrodes=None,
        scale_factor: float = 1e4,
        processed_dir: str = "processed_tusz_binary",
    ):
        assert mode in ["train", "val", "test"]
        self.scale_factor = scale_factor
        self.data_path = pjoin(path, processed_dir, mode)
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(
                f"TUSZ folder not found: {self.data_path} "
                f"(path={path!r}, processed_dir={processed_dir!r}, mode={mode!r})"
            )

        x_files = sorted(f for f in os.listdir(self.data_path) if f.endswith("_X.npy"))
        if not x_files:
            raise FileNotFoundError(f"No *_X.npy files found in {self.data_path}")
        self.x_paths = [pjoin(self.data_path, f) for f in x_files]

        # global index: sample i -> (recording file_ids[i], window win_ids[i])
        file_ids, win_ids, labels = [], [], []
        for fid, f in enumerate(x_files):
            y = np.load(pjoin(self.data_path, f.replace("_X.npy", "_y.npy")))
            file_ids.append(np.full(len(y), fid, dtype=np.int32))
            win_ids.append(np.arange(len(y), dtype=np.int32))
            labels.append(y)
        self.file_ids = np.concatenate(file_ids)
        self.win_ids = np.concatenate(win_ids)
        self.labels = np.concatenate(labels).astype(np.int64)

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes).float()

    def __len__(self):
        return len(self.labels)

    def class_counts(self):
        """Windows per class; handy for class weights or a WeightedRandomSampler."""
        return np.bincount(self.labels)

    def __getitem__(self, index):
        x = np.load(self.x_paths[int(self.file_ids[index])], mmap_mode="r")[int(self.win_ids[index])]
        X = torch.from_numpy(np.array(x))  # copy out of the memmap -> (C, T)
        Y = torch.tensor(self.labels[index])  # scalar

        return {
            "sample": X.float() * self.scale_factor,
            "label": Y.long(),
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([x["sample"] for x in batch]),
            "label": torch.stack([x["label"] for x in batch]),
            "pos": self.positions.unsqueeze(0).repeat(len(batch), 1, 1),
        }
