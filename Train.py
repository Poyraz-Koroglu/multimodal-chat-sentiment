import torch
from torch import nn
from torch import optim
from torch.utils.data import dataloader
from tqdm import tqdm

def train_model(model, train_loader, epochs, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct, total = 0, 0

        for batch in tqdm(train_loader):
            audio = batch['audio'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(audio, input_ids, attention_mask, labels)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch + 1}: Loss={epoch_loss / len(dataloader):.4f}, Acc={acc:.4f}")