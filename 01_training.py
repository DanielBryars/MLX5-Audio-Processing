import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import scipy.stats

from dataset import UrbanSoundDatasetFactory
from model import SoundClassifier

full_dataset = UrbanSoundDatasetFactory().CreateDataset()

# Example: Get all samples for fold 1
fold1_indices = [i for i, (_, _, fold) in enumerate(full_dataset) if fold == 1]
fold1_dataset = torch.utils.data.Subset(full_dataset, fold1_indices)

train_loader = DataLoader(fold1_dataset, batch_size=32, shuffle=True)


import torch.nn as nn
import torch.optim as optim

model = SoundClassifier(num_classes=10).to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0
    for waveforms, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")