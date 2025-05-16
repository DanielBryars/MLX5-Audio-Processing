from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os
import datetime
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import torch
from words import get_thing_explainer_vocab
from torch.optim import AdamW
#from AudioDataset import AudioDataset
from torch.utils.data import DataLoader
import torch
import torchaudio
import wandb
from model import *

ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

#Freeze encoder
model.model.encoder.requires_grad_(False)
for param in model.model.encoder.parameters():
    param.requires_grad = False

hyperparameters = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 4,
        'patience': 3,
        'num_epochs':10
}

wandb.init(project='MLX7-W5-AUDIO-001', config=hyperparameters)
config = wandb.config

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparameters["learning_rate"])

from torch.nn import CrossEntropyLoss

loss_fn = CrossEntropyLoss(ignore_index=-100)  # Padding tokens masked

from tqdm import tqdm

model.train()
model.to(device)

num_epochs = hyperparameters["num_epochs"]

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

if MONROE_ENGLISH_TOKEN not in processor.tokenizer.get_vocab():
    print(f"Adding token {MONROE_ENGLISH_TOKEN}")
    processor.tokenizer.add_tokens(["<|monroe|>"])
    model.resize_token_embeddings(len(processor.tokenizer))

script_dir = os.path.dirname(os.path.abspath(__file__))


train_dataset = AudioDataset(script_dir,"train_audio_dataset_results_300.json", processor)
val_dataset = AudioDataset(script_dir,"validation_audio_dataset_results_100.json", processor)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=hyperparameters["batch_size"],
    shuffle=True,
    collate_fn=whisper_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=hyperparameters["batch_size"],
    collate_fn=whisper_collate_fn
)

step = 0
best_val_loss = float('inf')
epochs_no_improve = 0
patience= hyperparameters['patience']
epoch_pbar = tqdm(range(1, hyperparameters['num_epochs'] + 1))

def save_checkpoint(model, hyperparameters, epoch, ts):
    checkpoint_dir = '/tmp'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_type = type(model).__name__
    descriptive_name = f'ts.{ts}.epoch.{epoch + 1}.{model_type}'
    checkpoint_name = f'{descriptive_name}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    #"/tmp/whisper_monroe.pt"

    print(f"Saving '{checkpoint_path}'")
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'hyperparameters': hyperparameters
    }, checkpoint_path)

    # Create wandb artifact and log it
    artifact = wandb.Artifact(
        name=descriptive_name,
        type='model',
        description=f'{model_type} model weights from epoch {epoch + 1}, timestamp {ts}')
    
    #actually upload the artifact!!!!
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)

for epoch in epoch_pbar:
    total_loss = 0.0

    loop = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:
        
        input_features = batch["input_features"].to(device)  # Already preprocessed
        labels = batch["labels"].to(device)

        # 3. Forward + Loss
        outputs = model(input_features=input_features, labels=labels)

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        wandb.log({'train/loss': loss.item()}, step=step)
        loop.set_postfix(loss=f"{loss.item():.4f}")
        step += 1

    avg_loss = total_loss / len(train_dataloader)
    wandb.log({'train/avg_loss': avg_loss}, step=step)

    save_checkpoint(model, hyperparameters, epoch, ts)


    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_features=input_features, labels=labels)
            val_loss += outputs.loss.item()

            decoded_ids = torch.argmax(outputs.logits, dim=-1)
            decoded_texts = processor.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)

            if epoch == 1 or epoch == 5:  # or any other condition to limit frequency
                table = wandb.Table(columns=["Epoch", "Index", "Original", "Predicted"])
                for i in range(min(10, len(decoded_texts))):
                    table.add_data(epoch, i, batch["original_text"][i], decoded_texts[i])
                    wandb.log({"val/samples": table}, step=step)


    val_loss /= len(val_loader)
    epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}", val_loss=f"{val_loss:.4f}")
    model.train()

    wandb.log({'val/avg_loss': avg_loss}, step=step)