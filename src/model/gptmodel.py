from transformers import XLMRobertaModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv
import json
from transformers import XLMRobertaTokenizerFast
import ast 

tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

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
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row in reader:
            author, timestamp, subreddit, language, text, labels = row 
            labels = ast.literal_eval(labels)

            texts.append(text)
            tags.append(LABEL_IDS[label] for label in labels)
    return texts, tags




class NERSentimentData(Dataset):
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]

        tokenized_inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

        input_ids = tokenized_inputs['input_ids'][0] 
        attention_mask = tokenized_inputs['attention_mask'][0] 


        label_ids = [-100] * len(input_ids)  # Initialize with -100 to ignore padding in loss calculation

        word_ids = tokenized_inputs.word_ids(batch_index=0)
        previous_word_id = None 


        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != previous_word_id:
                    if word_id < len(tags):
                        label = tags[word_id]
                        label_ids[i] = LABEL_IDS.get(label, LABEL_IDS['O'])
                previous_word_id = word_id

            

        return input_ids, attention_mask, torch.tensor(label_ids, dtype=torch.long)




class MultitaskModel(nn.Module):
    def __init__(self, model_name, num_ner_tags, num_sentiment_tags):
        super(MultitaskModel, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.xlmr.config.hidden_size
        self.ner_classifier = nn.Linear(hidden_size, num_ner_tags)
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_tags)

    def forward(self, input_ids, attention_mask, task_name):
        outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if task_name == 'ner':
            ner_logits = self.ner_classifier(sequence_output)
            return ner_logits
        elif task_name == 'sentiment':
            sentiment_logits = self.sentiment_classifier(pooled_output)
            return sentiment_logits

def compute_normalized_gradients(model, loss_ner, loss_sentiment, optimizer):
    optimizer.zero_grad()
    gradients_ner = torch.autograd.grad(loss_ner, model.parameters(), retain_graph=True)
    gradients_sentiment = torch.autograd.grad(loss_sentiment, model.parameters(), retain_graph=True)

    norm_ner = torch.sqrt(sum((g ** 2).sum() if g is not None else 0 for g in gradients_ner))
    norm_sentiment = torch.sqrt(sum((g ** 2).sum() if g is not None else 0 for g in gradients_sentiment))

    for (g_ner, g_sentiment, param) in zip(gradients_ner, gradients_sentiment, model.parameters()):
        normalized_g_ner = g_ner / norm_ner if norm_ner > 0 else 0
        normalized_g_sentiment = g_sentiment / norm_sentiment if norm_sentiment > 0 else 0
        param.grad = (normalized_g_ner + normalized_g_sentiment) / 2
    optimizer.step()

def train_model(model, dataloader, optimizer, loss_fn_ner, loss_fn_sentiment, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()


            ner_outputs = model(input_ids, attention_mask, 'ner')
            sentiment_outputs = model(input_ids, attention_mask, 'sentiment')
            num_classes_ner = ner_outputs.size(-1)
            num_classes_sentiment = sentiment_outputs.size(-1)

            ner_loss = loss_fn_ner(ner_outputs.view(-1, num_classes_ner), labels.view(-1))
            sentiment_loss = loss_fn_sentiment(sentiment_outputs.view(-1, num_classes_sentiment), labels.view(-1))

            loss = ner_loss + sentiment_loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs} completed.")
            print(f"Loss: {loss.item()}")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    texts, tags = process_data('cleaned_data.csv')
    dataset = NERSentimentData(texts, tags)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # Model initialization
    model_name = 'xlm-roberta-base'  # Use the appropriate model name
    num_ner_tags = 25  # Set the number of NER tags according to your dataset
    num_sentiment_tags = 3  # Set the number of sentiment classes according to your dataset
    model = MultitaskModel(model_name, num_ner_tags, num_sentiment_tags).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Assume train_dataloaders is defined elsewhere
    # Assume loss_fn_ner and loss_fn_sentiment are defined appropriately
    loss_fn_ner = nn.CrossEntropyLoss(ignore_index=-100) 
    
    loss_fn_sentiment = nn.CrossEntropyLoss(ignore_index=-100)

    train_model(model, dataloader, optimizer, loss_fn_ner, loss_fn_sentiment, num_epochs=10, device=device)
    torch.save(model.state_dict(), 'multitask_model_state_dict.pt')


if __name__ == '__main__':
    main()

