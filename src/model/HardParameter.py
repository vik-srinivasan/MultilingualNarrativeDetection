from transformers import XLMRobertaModel, XLMRobertaConfig
import torch.nn as nn 
import torch


class MultitaskModel(nn.Module):
    def __init__(self, ner_model_name, num_labels, sentiment_model_name, num_ner_tags, num_sentiment_tags):
        super(MultitaskModel, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained('xlm-roberta-base')

        self.xlmr_ner = XLMRobertaModel.from_pretrained(ner_model_name) #ner model
        self.xlmr_sentiment = XLMRobertaModel.from_pretrained(sentiment_model_name) #sentiment model

        self.ner_classifier = nn.Linear(self.xlmr_ner.hidden_size, num_ner_tags)
        self.sentiment_classifier = nn.Linear(self.xlmr_sentiment.hidden_size, num_sentiment_tags)


        
    def forward(self, input_ids, attention_mask, task_name):
        if task_name == 'ner':
            outputs = self.xlmr_ner(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs[0]
            ner_logits = self.ner_classifier(sequence_output)
            return ner_logits
        elif task_name == 'sentiment':
            outputs = self.xlmr_sentiment(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            sentiment_logits = self.sentiment_classifier(pooled_output)
            return sentiment_logits



    
    def compute_normalized_gradients(model, loss_ner, loss_sentiment, optimizer):
            # Compute gradients for each task
        optimizer.zero_grad()
        gradients_ner = torch.autograd.grad(loss_ner, model.parameters(), retain_graph=True, create_graph=True)
        optimizer.zero_grad()
        gradients_sentiment = torch.autograd.grad(loss_sentiment, model.parameters(), create_graph=True)

        # Normalize gradients
        norm_ner = torch.sqrt(sum((g ** 2).sum() for g in gradients_ner))
        norm_sentiment = torch.sqrt(sum((g ** 2).sum() for g in gradients_sentiment))

        # Apply normalized gradients
        for (g_ner, g_sentiment, param) in zip(gradients_ner, gradients_sentiment, model.parameters()):
            # Avoid dividing by zero
            normalized_g_ner = g_ner / norm_ner if norm_ner > 0 else g_ner
            normalized_g_sentiment = g_sentiment / norm_sentiment if norm_sentiment > 0 else g_sentiment
            # Assign the average of normalized gradients
            param.grad = (normalized_g_ner + normalized_g_sentiment) / 2

        # Perform the optimizer step outside of this function for clarity
        optimizer.step()
    
    def train_model(self, model, train_dataloaders, optimizer, num_epochs):
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
                self.compute_normalized_gradients(model, loss_ner, loss_sentiment, optimizer)

                print(f"Epoch {epoch+1}/{num_epochs} completed.")
        return model
    


    