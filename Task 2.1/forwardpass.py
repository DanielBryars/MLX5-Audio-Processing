import wandb
from model import *

ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

#if MONROE_ENGLISH_TOKEN not in processor.tokenizer.get_vocab():
#    print(f"Adding token {MONROE_ENGLISH_TOKEN}")
#    processor.tokenizer.add_tokens(["<|monroe|>"])
#    model.resize_token_embeddings(len(processor.tokenizer))

script_dir = os.path.dirname(os.path.abspath(__file__))

train_dataset = AudioDataset(script_dir,"train_audio_dataset_results_300.json", processor)
val_dataset = AudioDataset(script_dir,"validation_audio_dataset_results_100.json", processor)


batch_size = 1
dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=whisper_collate_fn
)

#wandb.init(project='MLX7-W5-AUDIO-FORWARD-PASS')

model.eval()

with torch.no_grad():
    for batch in dataloader:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        original_texts = batch["original_text"]
        audio_paths = batch["audio_path"]

        outputs = model(input_features=input_features, labels=labels)
        
        decoded_ids = torch.argmax(outputs.logits, dim=-1)
        decoded_texts = processor.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)

        print (audio_paths)
        print (original_texts)
        print (decoded_texts)

        break
        #if epoch == 1 or epoch == 5:  # or any other condition to limit frequency
        #    table = wandb.Table(columns=["Epoch", "Index", "Original", "Predicted"])
        #    for i in range(min(10, len(decoded_texts))):
        #        table.add_data(epoch, i, batch["original_text"][i], decoded_texts[i])
        #        wandb.log({"val/samples": table}, step=step)
