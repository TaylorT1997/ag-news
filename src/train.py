import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
from functools import partial

from dataset import AGNewsDataset

import sys
import math
import wandb


def collate_fn(batch):
    titles, descriptions, labels = [], [], []
    for title, description, label in batch:
        titles.append(title)
        descriptions.append(description)
        labels.append(label)
    return titles, descriptions, labels


num_epochs = 10
batch_size = 8
lr = 2e-5
device = "cuda"
use_wandb = True
testing = True


if use_wandb:
    wandb.init(project="ag-news", entity="taylort1997")
    config = wandb.config
    config.num_epochs = num_epochs
    config.batch_size = batch_size
    config.lr = lr

model_config = BertConfig.from_pretrained("bert-base-cased", num_labels=4)
model = BertForSequenceClassification(model_config)

model.to(device)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

train_dataset = AGNewsDataset(mode="train")
test_dataset = AGNewsDataset(mode="test")

dataset_len = len(train_dataset)
num_train_samples = math.ceil(dataset_len * 0.8)
num_val_samples = math.floor(dataset_len * 0.2)
train_dataset, val_dataset = random_split(
    train_dataset, [num_train_samples, num_val_samples]
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
)
val_loader = DataLoader(val_dataset, batch_size=batch_size * 4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=lr)

for epoch in range(num_epochs):

    # Train loop
    model.train()

    total_corrects = 0
    total_samples = 0
    total_loss = 0

    for idx, (titles, descriptions, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        tokenized_sequences = tokenizer(
            titles, descriptions, padding=True, truncation=True, return_tensors="pt"
        )

        input_ids = tokenized_sequences["input_ids"].to(device)
        attention_masks = tokenized_sequences["attention_mask"].to(device)
        labels = torch.tensor(labels, device=device)

        outputs = model(input_ids, attention_mask=attention_masks, labels=labels,)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        corrects = torch.sum(labels == preds)

        total_corrects += corrects
        total_loss += loss.item()
        total_samples += len(labels)

        if use_wandb:
            wandb.log({"train_loss": loss})

    if use_wandb:
        wandb.log({"train_acc": total_corrects / total_samples})

    print(f"Epoch: {epoch+1}/{num_epochs}")
    print()
    print(f"Training loss: {total_loss/total_samples}")
    print(f"Training acc: {total_corrects/total_samples}")
    print()

    # Validation loop
    model.eval()

    total_corrects = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for idx, (titles, descriptions, labels) in enumerate(val_loader):
            optimizer.zero_grad()

            tokenized_sequences = tokenizer(
                titles, descriptions, padding=True, truncation=True, return_tensors="pt"
            )

            input_ids = tokenized_sequences["input_ids"].to(device)
            attention_masks = tokenized_sequences["attention_mask"].to(device)
            labels = torch.tensor(labels, device=device)

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels,)

            loss = outputs.loss

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            corrects = torch.sum(labels == preds)

            total_corrects += corrects
            total_loss += loss.item()
            total_samples += len(labels)

            if use_wandb:
                wandb.log({"val_loss": loss})

        if use_wandb:
            wandb.log({"val_acc": total_corrects / total_samples})

        print(f"Validation loss: {total_loss/total_samples}")
        print(f"Validation acc: {total_corrects/total_samples}")
        print()
        print()

# Testing procedure
if testing:
    total_corrects = 0
    total_samples = 0
    for idx, (titles, descriptions, labels) in enumerate(test_loader):

        tokenized_sequences = tokenizer(
            titles, descriptions, padding=True, truncation=True, return_tensors="pt"
        )

        input_ids = tokenized_sequences["input_ids"].to(device)
        attention_masks = tokenized_sequences["attention_mask"].to(device)
        labels = torch.tensor(labels, device=device)

        outputs = model(input_ids, attention_mask=attention_masks, labels=labels,)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        corrects = torch.sum(labels == preds)

        total_corrects += corrects
        total_samples += len(labels)

    print(f"Testing accuracy: {total_corrects/total_samples}")

