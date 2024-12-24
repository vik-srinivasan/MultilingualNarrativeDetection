import numpy as np
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from newtry2 import tokenizer, NERSentimentDataset, MultitaskModel, process_data  # Adjust the import based on your actual structure

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
# Load your custom model
model_path = "model.pth"
model = MultitaskModel('xlm-roberta-base', len(LABEL_IDS))
model.load_state_dict(torch.load(model_path))
model.eval()

# Load and process your dataset
texts, tags = process_data('full_cleaned_data.csv')
dataset = NERSentimentDataset(texts, tags)

# Split dataset into train and test
small_dataset = dataset[:20]  # Select only 20 rows for this example
train_size = int(0.5 * len(small_dataset))
test_size = len(small_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(small_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the evaluation metric
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).astype(np.float32).mean().item()}

# Define the objective function for optuna
def objective(trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
    num_train_epochs = trial.suggest_int('num_train_epochs', 2, 4)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16])
    
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=num_train_epochs,  # number of training epochs
        per_device_train_batch_size=per_device_train_batch_size,  # batch size for training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        learning_rate=learning_rate,  # learning rate
        evaluation_strategy="epoch",  # Evaluate every epoch
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # evaluation metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    accuracy = eval_results['eval_accuracy']
    return accuracy

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Print the best hyperparameters
print(f"Best Hyperparameters: {study.best_params}")
print(f"Best Accuracy: {study.best_value}")

# Save the best model
best_trial = study.best_trial
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=best_trial.params['num_train_epochs'],  # number of training epochs
    per_device_train_batch_size=best_trial.params['per_device_train_batch_size'],  # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
    learning_rate=best_trial.params['learning_rate'],  # learning rate
    evaluation_strategy="epoch",  # Evaluate every epoch
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # evaluation metrics
)

trainer.train()
trainer.save_model("best_model")
