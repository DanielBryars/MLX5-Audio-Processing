import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import scipy.stats

class UrbanSoundDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
        self.label2id = {label: i for i, label in enumerate(self.data.features["class"].names)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        x = torch.tensor(row["mfcc_summary"], dtype=torch.float32)
        y = torch.tensor(self.label2id[row["class"]], dtype=torch.long)
        fold = row["fold"]
        return x, y, fold

class UrbanSoundDatasetFactory:
    def CreateDataset(self):
        # Load the dataset
        ds = load_dataset("danavery/urbansound8K", split="train")

        # MFCC transform
        mfcc_transform = T.MFCC(
            sample_rate=44100,
            n_mfcc=25,
            melkwargs={
                'n_fft': 1024,
                'hop_length': 512,
                'n_mels': 40,
                'f_min': 0.0,
                'f_max': 22050
            }
        )

        # Compute MFCCs
        def extract_mfcc(batch):
            audio = batch["audio"]["array"]
            sr = batch["audio"]["sampling_rate"]
            if sr != 44100:
                raise ValueError("Unexpected sampling rate")

            waveform = torch.tensor(audio).unsqueeze(0)
            mfcc = mfcc_transform(waveform).squeeze(0)  # shape: [25, T]
            batch["mfcc"] = mfcc.numpy()
            return batch

        ds = ds.map(extract_mfcc)

        # Summarise MFCCs
        def summarise_mfcc(batch):
            x = batch["mfcc"]  # shape: [25, T]
            delta1 = F.compute_deltas(torch.tensor(x))
            delta2 = F.compute_deltas(delta1)

            def stats(t):
                return torch.stack([
                    t.min(dim=1).values,
                    t.max(dim=1).values,
                    t.median(dim=1).values,
                    t.mean(dim=1),
                    t.var(dim=1),
                    torch.tensor(scipy.stats.skew(t.numpy(), axis=1)),
                    torch.tensor(scipy.stats.kurtosis(t.numpy(), axis=1))
                ])

            features = torch.cat([
                stats(torch.tensor(x)),
                delta1.mean(dim=1, keepdim=True).T,
                delta1.var(dim=1, keepdim=True).T,
                delta2.mean(dim=1, keepdim=True).T,
                delta2.var(dim=1, keepdim=True).T
            ], dim=0)  # shape: [9, 25] → flatten → [225]

            batch["mfcc_summary"] = features.flatten().numpy()
            return batch

        ds = ds.map(summarise_mfcc)

        return UrbanSoundDataset(ds)