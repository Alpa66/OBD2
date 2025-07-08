import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class SlidingWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target: pd.Series, window_size: int, stride: int = 1, get_step_next = False):
        # Convert data and targets to numpy 
        self.features = list(df.columns)
        self.data = torch.as_tensor(df.values, dtype=torch.float32)
        self.target = torch.as_tensor(target.values, dtype=torch.float32) if target is not None else None
        
        # Save multindex for safety checks
        self.index = df.index
        self.window_size = window_size
        self.get_step_next = get_step_next
        if self.get_step_next:
            self.window_size += 1
        # Calculate valid windows for each run_id
        self.valid_windows = self._precompute_valid_windows(stride)

    def _precompute_valid_windows(self, stride):
        valid_windows = []
        run_ids = self.index.get_level_values(0).unique()

        for run_id in tqdm(run_ids, desc="Building safe windows"):
            # Get all indices for running run_id
            run_mask = self.index.get_level_values(0) == run_id
            run_indices = np.where(run_mask)[0]
        
            # Check run_id has enough points
            if len(run_indices) < self.window_size:
                continue
                
            # Generate end window indices ONLY in the borders of this run_id
            for end_pos in range(self.window_size, len(run_indices), stride):
                start_pos = end_pos - self.window_size
                start_idx = run_indices[start_pos]
                end_idx = run_indices[end_pos]
                
                # Extra check
                assert self.index[start_idx][0] == self.index[end_idx-1][0], "Window crosses run_id boundary!"
                
                valid_windows.append((start_idx, end_idx))
                
        return valid_windows

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        start_idx, end_idx = self.valid_windows[idx]
        sample = self.data[start_idx:end_idx]
        target = self.target[start_idx:end_idx].max() if self.target is not None else sample[-1]
        return sample, target
