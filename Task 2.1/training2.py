import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import wandb
from tqdm import tqdm

def train(model, processor, train_dataset, val_dataset, training_args):
    train_dataloader = DataLoader(train_dataset, batch_size=training_args['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=training_args['batch_size'])

    # Separate learning rates for encoder and decoder
    optimizer = AdamW([
        {"params": model.model.encoder.parameters(), "lr": training_args['learning_rate'] * 0.1},
        {"params": model.model.decoder.parameters(), "lr": training_args['learning_rate']}
    ])

    num_training_steps = training_args['num_epochs'] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wandb.init(project="whisper-finetune", config=training_args)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(training_args['num_epochs']):
        print(f"Epoch {epoch + 1}/{training_args['num_epochs']}")

        # Train
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            decoder_input_ids = labels[:, :-1].contiguous()
            label_ids = labels[:, 1:].contiguous()

            outputs = model(input_features=input_features, decoder_input_ids=decoder_input_ids, labels=label_ids)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validate
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                decoder_input_ids = labels[:, :-1].contiguous()
                label_ids = labels[:, 1:].contiguous()

                outputs = model(input_features=input_features, decoder_input_ids=decoder_input_ids, labels=label_ids)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), training_args['save_path'])
            print("Model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_args['patience']:
                print("Early stopping triggered.")
                break