import json
from typing import List, Dict
from torch.utils.data import Dataset


class ReportsDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Expect {"prompt": "...", "reference": "..."}
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
