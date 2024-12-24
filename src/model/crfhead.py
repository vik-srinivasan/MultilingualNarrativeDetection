from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import csv
import ast
from TorchCRF import CRF


# Tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

# Constants
NER_LABEL_IDS = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6,
    'B-TIME': 7,
    'I-TIME': 8,
    'B-O': 0,
    'I-O': 0,
}
SENTIMENT_LABEL_IDS = {
    'neutral': 0,
    'positive': 1,
    'negative': 2,
}

def process_data(input_file):
    texts, ner_tags, sentiment_tags = [], [], []
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            author, timestamp, subreddit, language, text, label_str = row
            text = row[4]
            labels = label_str.strip("[]").split(", ")
            entities = [label.split('-')[0] + '-' + label.split('-')[1] if '-' in label else 'O' for label in labels]
            sentiments = [label.split('-')[2] if '-' in label else 'neutral' for label in labels]
            processed_entities = [NER_LABEL_IDS.get(entity, 0) for entity in entities]
            processed_sentiments = [SENTIMENT_LABEL_IDS.get(sentiment, 0) for sentiment in sentiments]
            texts.append(text)
            ner_tags.append(processed_entities)
            sentiment_tags.append(processed_sentiments)
    return texts, ner_tags, sentiment_tags 

class NERSentimentDataset(Dataset):
    def __init__(self, texts, ner_tags, sentiment_tags):
        self.texts = texts
        self.ner_tags = ner_tags
        self.sentiment_tags = sentiment_tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        entities = self.ner_tags[idx]
        sentiments = self.sentiment_tags[idx]
        
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Ensure labels match input_ids length
        entity_ids = torch.full((128,), -100, dtype=torch.long)
        sentiment_ids = torch.full((128,), -100, dtype=torch.long)
        entity_tensor = torch.tensor(entities, dtype=torch.long)
        sentiment_tensor = torch.tensor(sentiments, dtype=torch.long)
        truncated_entities = entity_tensor[:128]
        truncated_sentiments = sentiment_tensor[:128]
        entity_ids[:len(truncated_entities)] = truncated_entities
        sentiment_ids[:len(truncated_sentiments)] = truncated_sentiments

        sequence_sentiment = torch.mean(sentiment_ids[sentiment_ids != -100].float()).round().long()



        return input_ids, attention_mask, entity_ids, sequence_sentiment

class MultitaskModel(nn.Module):
    def __init__(self, model_name, num_labels_ner, num_labels_sentiment,dropout_rate=0.1, freeze_layers = 6):
        super().__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.sentiment_classifier = nn.Linear(self.xlmr.config.hidden_size, num_labels_sentiment)
        self.crf = CRF(num_labels_ner)

        if freeze_layers > 0:
            for param in self.xlmr.embeddings.parameters():
                param.requires_grad = False
            for layer in self.xlmr.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, task_name, labels=None):
        outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])

        if task_name == 'ner':
            ner_logits = sequence_output
            if labels is not None:
                mask = (labels != -100)
                loss = -self.crf(ner_logits, labels, mask=attention_mask.byte()).mean()
                return loss
            else:
                return self.crf.decode(ner_logits, mask=attention_mask.byte())
        elif task_name == 'sentiment':
            pooled_output = self.dropout(outputs[1])

            sentiment_logits = self.sentiment_classifier(pooled_output)
            return sentiment_logits
        else:
            raise ValueError(f'Invalid task name: {task_name}')


def train_model(model, dataloader, optimizer, loss_fn_ner, loss_fn_sentiment, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for input_ids, attention_mask, entity_ids, sentiment_ids in dataloader:
            input_ids, attention_mask, entity_ids, sentiment_ids = input_ids.to(device), attention_mask.to(device), entity_ids.to(device), sentiment_ids.to(device)
            optimizer.zero_grad()
            
            ner_loss = model(input_ids, attention_mask, 'ner', entity_ids)

            outputs_sentiment = model(input_ids, attention_mask, 'sentiment')
            sentiment_loss = loss_fn_sentiment(outputs_sentiment, sentiment_ids) 
            loss = ner_loss + sentiment_loss
            loss.backward()
            optimizer.step()
            print('sentiloss:',sentiment_loss)
            print('nerloss:', ner_loss)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, NER Loss: {ner_loss.item()}, Sentiment Loss: {sentiment_loss.item()}')




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_labels_ner = len(NER_LABEL_IDS)
num_labels_sentiment = len(SENTIMENT_LABEL_IDS)
model = MultitaskModel('xlm-roberta-base', num_labels_ner, num_labels_sentiment).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
loss_fn_sentiment = nn.CrossEntropyLoss(ignore_index=-100)
loss_fn_ner = nn.CrossEntropyLoss(ignore_index=-100)

texts, ner_tags, sentiment_tags = process_data('cleaned_data.csv')
dataset = NERSentimentDataset(texts, ner_tags, sentiment_tags)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


train_model(model, dataloader, optimizer, loss_fn_ner, loss_fn_sentiment, num_epochs=20, device=device)


torch.save(model.state_dict(), 'model.pth')


