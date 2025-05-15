from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os

from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import torch
from words import get_thing_explainer_vocab
from torch.optim import AdamW


#{
#    "input_features": whisper_processor.feature_extractor(audio, sampling_rate=16000).input_features[0],  # [80, 3000]
#    "labels": whisper_processor.tokenizer(
#        "<|startoftranscript|><|monroe|><|notimestamps|>" + simplified_text + "<|endoftranscript|>",
#        return_tensors="pt"
#    ).input_ids[0]
#}


processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")



#Freeze encoder
model.model.encoder.requires_grad_(False)
for param in model.model.encoder.parameters():
    param.requires_grad = False



hyperparameters = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 64,
        'patience': 3,
        'num_layers': 4,
        'num_heads':2,
        'dropout':0.1,
        'num_epochs':10
}

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparameters["learning_rate"])

from torch.nn import CrossEntropyLoss

loss_fn = CrossEntropyLoss(ignore_index=-100)  # Padding tokens masked

from tqdm import tqdm

model.train()
model.to("cuda")

num_epochs = hyperparameters["num_epochs"]

for epoch in tqdm(range(num_epochs)):
    total_loss = 0.0

    for batch in tqdm(dataloader):
        input_features = batch["input_features"].to("cuda")         # [B, 80, 3000]
        labels = batch["labels"].to("cuda")                         # [B, T] with -100 for padding

        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader):.4f}")