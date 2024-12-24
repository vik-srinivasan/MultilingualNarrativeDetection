from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import csv
import ast

# Tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

# Constants
LABEL_IDS = {
    'O': 0,
    'B-PER-neutral': 1,
    'B-PER-positive': 2,
    'B-PER-negative': 3,
    'I-PER-neutral': 4,
    'I-PER-positive': 5,
    'I-PER-negative': 6,
    'B-ORG-neutral': 7,
    'B-ORG-positive': 8,
    'B-ORG-negative': 9,
    'I-ORG-neutral': 10,
    'I-ORG-positive': 11,
    'I-ORG-negative': 12,
    'B-LOC-neutral': 13,
    'B-LOC-positive': 14,
    'B-LOC-negative': 15,
    'I-LOC-neutral': 16,
    'I-LOC-positive': 17,
    'I-LOC-negative': 18,
    'B-TIME-neutral': 19,
    'B-TIME-positive': 20,
    'B-TIME-negative': 21,
    'I-TIME-neutral': 22,
    'I-TIME-positive': 23,
    'I-TIME-negative': 24,


    'B-O-neutral': 0,
    'B-O-positive': 0,
    'B-O-negative': 0,
    'I-O-neutral': 0,
    'I-O-positive': 0,
    'I-O-negative': 0,

}

def process_data(input_file):
    texts, tags = [], []
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            author, timestamp, subreddit, language, text, label_str = row
            text = row[4]
            labels = ast.literal_eval(row[5])
            texts.append(text)
            labels = label_str.strip("[]").split(", ")
            labels = [label.strip("'") for label in labels]  # Clean up any extra quotes
            processed_labels = [LABEL_IDS.get(label, 0) for label in labels]
            tags.append(processed_labels)
    return texts, tags

class NERSentimentDataset(Dataset):
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.tags[idx]
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Ensure labels match input_ids length
        label_ids = torch.full((128,), -100, dtype=torch.long)
        label_ids[:len(labels)] = torch.tensor(labels, dtype=torch.long)

        return input_ids, attention_mask, label_ids

class MultitaskModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.xlmr.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs[0])
        return logits

def train_model(model, dataloader, optimizer, loss_fn, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs.view(-1, model.classifier.out_features), labels.view(-1))
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultitaskModel('xlm-roberta-base', len(LABEL_IDS)).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

texts, tags = process_data('cleaned_data.csv')
dataset = NERSentimentDataset(texts, tags)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training
train_model(model, dataloader, optimizer, loss_fn, num_epochs=10, device=device)

# Save Model
torch.save(model.state_dict(), 'model.pth')
