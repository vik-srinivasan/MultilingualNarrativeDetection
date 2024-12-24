import numpy as np
import pandas as pd
import optuna
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from datasets import Dataset

# Load the sentiment analysis model and tokenizer
sentiment_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_config = AutoConfig.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Load your dataset
# Assume the CSV has columns 'Text' and 'Sentiment'
df = pd.read_csv('results.csv')
df = df[['Text', 'Sentiment']]  # Ensure only the required columns are loaded

# Convert the sentiment labels to numerical values if needed
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['Sentiment'] = df['Sentiment'].map(label_mapping)

# Convert the DataFrame to a Dataset
dataset = Dataset.from_pandas(df)

# Slice the dataset to only include 20 rows
small_dataset = dataset.select(range(20))

# Split dataset into train and test (10 each)
train_test_split = small_dataset.train_test_split(test_size=0.5)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Tokenize the dataset
def tokenize_function(examples):
    return sentiment_tokenizer(examples['Text'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Rename the 'Sentiment' column to 'labels' to match the expected input
train_dataset = train_dataset.rename_column('Sentiment', 'labels')
test_dataset = test_dataset.rename_column('Sentiment', 'labels')

# Set the format of the dataset to return PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define the evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': (predictions == labels).astype(np.float32).mean().item(),
    }

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
        model=sentiment_model,  # the instantiated 🤗 Transformers model to be trained
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
    model=sentiment_model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # evaluation metrics
)

trainer.train()
trainer.save_model("best_model")
