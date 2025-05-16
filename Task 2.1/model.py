from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os
import datetime
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import torch
from words import get_thing_explainer_vocab
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import json
import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

def mel2wandbimage(features, caption):

    nonzero_cols = np.where(features.sum(axis=0) != 0)[0]

    if len(nonzero_cols) > 0:
        features = features[:, :nonzero_cols[-1] + 1]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(features, origin='lower', aspect='auto', interpolation='nearest')
    ax.set_title("Log-Mel Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Frequency Bin")
    fig.colorbar(im, ax=ax)

    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    pil_image = Image.open(buf)
    wandb_image = wandb.Image(pil_image, caption=caption)
    return wandb_image


def whisper_collate_fn(batch):
    # Pad input_features (mel spectrograms) to 3000 frames
    padded_inputs = []
    for item in batch:
        feat = item["input_features"]  # [80, T]
        T = feat.shape[1]
        if T < 3000:
            pad_width = 3000 - T
            feat = torch.nn.functional.pad(feat, (0, pad_width))  # pad on right
        else:
            feat = feat[:, :3000]  # truncate if longer
        padded_inputs.append(feat)

    input_features = torch.stack(padded_inputs)  # [B, 80, 3000]

    # Pad labels
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)

    return {
        "input_features": input_features,
        "labels": labels,
        "simplified_text": [item["simplified_text"] for item in batch], 
        "original_text": [item["original_text"] for item in batch],
        "audio_path": [item["audio_path"] for item in batch],
    }

def decode_outputs(logits, tokenizer, stop_token="<|endoftranscript|>"):
    """
    Decodes logits from Whisper into strings, truncates at <|endoftranscript|>, and cleans up any spillover.
    """
    # Greedy decoding (can be replaced with beam if needed)
    token_ids = torch.argmax(logits, dim=-1)

    # Decode into text
    decoded_texts = tokenizer.batch_decode(token_ids, skip_special_tokens=False)

    cleaned_texts = []
    for text in decoded_texts:
        # Truncate at the stop token if present
        if stop_token in text:
            text = text.split(stop_token)[0]
        
        # Remove any stray trailing characters
        text = text.strip().replace("<|startoftranscript|>", "").replace("<|notimestamps|>", "").replace("<|monroe|>", "")
        cleaned_texts.append(text.strip())

    return cleaned_texts

MONROE_ENGLISH_TOKEN = "<|monroe|>"

class AudioDataset(Dataset):
    def __init__(self, basepath, json_path, processor):
        """
        :param json_path: Path to the JSON file (e.g. train_audio_dataset_results_300.json)
        :param processor: A processor object (e.g. WhisperProcessor) that handles audio + text
        """

        self.basepath = basepath

        with open(os.path.join(basepath,json_path), 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        audio_path = os.path.join(self.basepath ,entry["audio_path"])
        simplified_text = entry["simplified_text"] #monroe simple english

        text = f"<|startoftranscript|>{MONROE_ENGLISH_TOKEN}<|notimestamps|> {simplified_text} <|endoftranscript|>"

        # Load audio (mono)
        waveform, sr = torchaudio.load(audio_path)
        
        #make sure it's a 16K
        nsr = 16000
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=nsr)
        waveform = resampler(waveform)

        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if stereo

        # Preprocess using the provided processor (e.g. WhisperProcessor)
        inputs = self.processor(
            audio=waveform.squeeze().numpy(), 
            sampling_rate=nsr,
            text=text, 
            return_tensors="pt", 
            padding=True
        )

        # Flatten batch dimension since DataLoader will re-batch
        item = {k: v.squeeze(0) for k, v in inputs.items()}



        item["simplified_text"] = simplified_text
        item["labels"][item["labels"] == self.processor.tokenizer.pad_token_id] = -100

        item["original_text"] = entry["original_text"]

        item["audio_path"] = audio_path
        return item
