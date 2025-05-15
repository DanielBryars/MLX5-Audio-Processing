import json
import torch
from torch.utils.data import Dataset
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, json_path, processor):
        """
        :param json_path: Path to the JSON file (e.g. train_audio_dataset_results_300.json)
        :param processor: A processor object (e.g. WhisperProcessor) that handles audio + text
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        audio_path = entry["audio_path"]
        original_text = entry["original_text"] #english 
        text = entry["simplified_text"] #monroe simple english

        # Load audio (mono)
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if stereo

        # Preprocess using the provided processor (e.g. WhisperProcessor)
        inputs = self.processor(
            audio=waveform.squeeze().numpy(), sampling_rate=sr,
            text=text, return_tensors="pt", padding=True
        )

        # Flatten batch dimension since DataLoader will re-batch
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        return item