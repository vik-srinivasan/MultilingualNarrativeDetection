from transformers import XLMRobertaModel
import torch
import torch.nn as nn
import torch.optim as optim

class MultitaskModel(nn.Module):
    def __init__(self, ner_model_name, sentiment_model_name, num_ner_tags, num_sentiment_tags):
        super(MultitaskModel, self).__init__()
        # Load separate pre-trained models for each task
        self.ner_model = XLMRobertaModel.from_pretrained(ner_model_name)
        self.sentiment_model = XLMRobertaModel.from_pretrained(sentiment_model_name)
        
        # Define the task-specific classifiers
        ner_hidden_size = self.ner_model.config.hidden_size
        sentiment_hidden_size = self.sentiment_model.config.hidden_size
        self.ner_classifier = nn.Linear(ner_hidden_size, num_ner_tags)
        self.sentiment_classifier = nn.Linear(sentiment_hidden_size, num_sentiment_tags)

    def forward(self, input_ids, attention_mask, task_name):
        # Route input through the appropriate model based on task
        if task_name == 'ner':
            outputs = self.ner_model(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs[0]
            ner_logits = self.ner_classifier(sequence_output)
            return ner_logits
        elif task_name == 'sentiment':
            outputs = self.sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            sentiment_logits = self.sentiment_classifier(pooled_output)
            return sentiment_logits

def compute_normalized_gradients(model, loss_ner, loss_sentiment, optimizer):
    # Zero out gradients before calculation
    optimizer.zero_grad()
    # Calculate gradients for each task
    gradients_ner = torch.autograd.grad(loss_ner, model.parameters(), retain_graph=True)
    gradients_sentiment = torch.autograd.grad(loss_sentiment, model.parameters(), retain_graph=True)
    
    # Calculate the norms of the gradients
    norm_ner = torch.sqrt(sum((g ** 2).sum() if g is not None else 0 for g in gradients_ner))
    norm_sentiment = torch.sqrt(sum((g ** 2).sum() if g is not None else 0 for g in gradients_sentiment))
    
    # Apply normalized gradients
    for (g_ner, g_sentiment, param) in zip(gradients_ner, gradients_sentiment, model.parameters()):
        normalized_g_ner = g_ner / norm_ner if norm_ner > 0 else 0
        normalized_g_sentiment = g_sentiment / norm_sentiment if norm_sentiment > 0 else 0
        param.grad = (normalized_g_ner + normalized_g_sentiment) / 2
    # Update the model parameters
    optimizer.step()

def train_model(model, train_dataloaders, optimizer, loss_fn_ner, loss_fn_sentiment, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for data_ner, data_sentiment in zip(train_dataloaders['ner'], train_dataloaders['sentiment']):
            inputs_ner, labels_ner = data_ner['inputs'], data_ner['labels']
            inputs_sentiment, labels_sentiment = data_sentiment['inputs'], data_sentiment['labels']

            # Forward pass for each task
            outputs_ner = model(inputs_ner['input_ids'], inputs_ner['attention_mask'], 'ner')
            loss_ner = loss_fn_ner(outputs_ner, labels_ner)
            outputs_sentiment = model(inputs_sentiment['input_ids'], inputs_sentiment['attention_mask'], 'sentiment')
            loss_sentiment = loss_fn_sentiment(outputs_sentiment, labels_sentiment)

            # Compute and apply normalized gradients
            compute_normalized_gradients(model, loss_ner, loss_sentiment, optimizer)

            print(f"Epoch {epoch+1}/{num_epochs} completed.")
    return model

# Model initialization
ner_model_name = 'path/to/ner/model'  # Specify the path to the pre-trained NER model
sentiment_model_name = 'path/to/sentiment/model'  # Specify the path to the pre-trained sentiment model
num_ner_tags = 9  # Set the number of NER tags according to your dataset
num_sentiment_tags = 3  # Set the number of sentiment classes according to your dataset
model = MultitaskModel(ner_model_name, sentiment_model_name, num_ner_tags, num_sentiment_tags)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Assume train_dataloaders is defined elsewhere
# Assume loss_fn_ner and loss
loss_fn_ner = nn.CrossEntropyLoss(ignore_index=-1) 
loss_fn_sentiment = nn.CrossEntropyLoss()

train_model(model, train_dataloaders, optimizer, loss_fn_ner, loss_fn_sentiment, num_epochs=10)
