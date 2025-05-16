#import whisper
#model = whisper.load_model("base")
#result = model.transcribe("ShippingForecast.1stMin.ulaw.8k.wav")
#print(result["text"])

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os

from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import torch
from words import get_thing_explainer_vocab

import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

# Load Whisper model and tokenizer
model_name = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name)
from transformers.generation.logits_process import LogitsProcessor


# Load Thing Explainer vocabulary (lowercased)
vocab_words = get_thing_explainer_vocab()
 
# Map allowed tokens
allowed_token_ids = set()
for word in vocab_words:
    token_ids = tokenizer(word, add_special_tokens=False).input_ids
    if len(token_ids) == 1:
        allowed_token_ids.add(token_ids[0])

# Wrap generation with vocabulary mask
def generate_limited_vocab(input_features):
    outputs = model.generate(
        input_features,
        do_sample=False,
        logits_processor=[
            ConstrainedTokenLogitsProcessor(allowed_token_ids)
        ]
    )
    return outputs

# Logits processor to zero out disallowed tokens

class ConstrainedTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids):
        self.allowed_ids = allowed_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_ids:
            mask[:, token_id] = 0
        return scores + mask
    
# Load processor and model

script_dir = os.path.dirname(os.path.abspath(__file__))

#
#audio_path = os.path.join(script_dir, "ShippingForecast.1stMin.ulaw.8k.wav")

audio_path = os.path.join(script_dir, 'audio_dataset', "0ab16188d6d1292238430c986e1fb9bdcb8fdc2ab593555201f01e6c0e7d08d982c328883afc778914398a5fae5426de2fe123dec76995fb1d2414562ec12b92.wav")


processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.eval()

waveform, sr = torchaudio.load(audio_path)

print(f"Loaded '{audio_path}'")

print(f"{waveform.shape} @ {sr}")

if waveform.shape[0] > 1:
    print("Concerting to Mono")
    waveform = waveform.mean(dim=0)
    waveform = waveform.unsqueeze(0)

print(f"After Mono Conversion {waveform.shape} @ {sr}")

nf = 16000
resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=nf)
waveform = resampler(waveform)
print(f"After resampling {waveform.shape} @ {nf}")

#inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
inputs = processor(waveform.numpy(), sampling_rate=nf, return_tensors="pt")


logits_processors = []
#logits_processors.append(ConstrainedTokenLogitsProcessor(allowed_token_ids))

with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_features"], 
        logits_processor=logits_processors,
    max_new_tokens=128)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)


run = wandb.init(project='MLX7-W5-AUDIO-WHISPER-TEST')

features = inputs["input_features"][0].cpu().numpy()  # shape: (80, T)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(features, origin='lower', aspect='auto', interpolation='none')
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

examples = []

wandb_image = wandb.Image(pil_image, caption=f"{transcription[0]}")
examples.append(wandb_image)
#wandb.log({"mel_spectrogram": wandb_image})

run.log({"examples": examples})

print(transcription[0])
