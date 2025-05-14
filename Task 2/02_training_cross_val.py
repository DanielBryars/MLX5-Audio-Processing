from torch.utils.data import Subset
from sklearn.metrics import accuracy_score

full_dataset = UrbanSoundDatasetFactory().CreateDataset()
fold_indices = {k: [i for i, (_, _, fold) in enumerate(full_dataset) if fold == k] for k in range(1, 11)}

all_accuracies = []

for val_fold in range(1, 11):
    print(f"\n=== Fold {val_fold} ===")

    val_idx = fold_indices[val_fold]
    train_idx = [i for k, idxs in fold_indices.items() if k != val_fold for i in idxs]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SoundClassifier(num_classes=10).to('cuda')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms = waveforms.to('cuda')
            outputs = model(waveforms).cpu()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Fold {val_fold} accuracy: {acc:.4f}")
    all_accuracies.append(acc)

print(f"\nAverage Accuracy over 10 folds: {np.mean(all_accuracies):.4f}")
