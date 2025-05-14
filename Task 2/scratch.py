#git add . && git commit -m "WIP" && git push
from pprint import pprint
import torch


from datasets import load_dataset

ds = load_dataset("danavery/urbansound8K")

print("here")
pprint(ds["train"].features)


sample = ds["train"][0]

audio_array = sample["audio"]["array"]  # numpy array
sampling_rate = sample["audio"]["sampling_rate"]
label = sample["class"]
filename = sample["slice_file_name"]

print(f"Label: {label}, File: {filename}, Audio shape: {audio_array.shape}, Sampling rate: {sampling_rate}")

import torchaudio
import torchaudio.transforms as T

mfcc_transform = T.MFCC(
    sample_rate=44100,      # UrbanSound8K is 44.1 kHz
    n_mfcc=25,
    melkwargs={
        'n_fft': 1024,
        'hop_length': 512,
        'n_mels': 40,
        'f_min': 0.0,
        'f_max': 22050
    }
)

def extract_mfcc(batch):
    audio = batch["audio"]["array"]
    sr = batch["audio"]["sampling_rate"]
    if sr != 44100:
        raise ValueError("Unexpected sampling rate")

    waveform = torch.tensor(audio).unsqueeze(0)
    mfcc = mfcc_transform(waveform).squeeze(0)  # shape: (n_mfcc, time)
    batch["mfcc"] = mfcc
    return batch


ds_with_mfcc = ds.map(extract_mfcc)  


ds_dog = ds_with_mfcc.filter(lambda x: x["class"] == "dog_bark")

def summarise_mfcc(batch):
    mfcc = batch["mfcc"]
    # First and second derivative
    delta1 = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta1)

    stats = torch.cat([
        mfcc.min(dim=1).values,
        mfcc.max(dim=1).values,
        mfcc.median(dim=1).values,
        mfcc.mean(dim=1),
        mfcc.var(dim=1),
        mfcc.skew(dim=1),
        mfcc.kurtosis(dim=1),
        delta1.mean(dim=1),
        delta1.var(dim=1),
        delta2.mean(dim=1),
        delta2.var(dim=1)
    ])
    batch["mfcc_summary"] = stats
    return batch

ds_final = ds_with_mfcc.map(summarise_mfcc)


'''
import soundata

dataset = soundata.initialize('urbansound8k')
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data
'''