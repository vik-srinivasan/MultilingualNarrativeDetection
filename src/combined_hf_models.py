import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline, AutoConfig
import numpy as np
from scipy.special import softmax

# Load the sentiment analysis model and tokenizer
sentiment_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_config = AutoConfig.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Load the NER model and tokenizer
ner_model_name = "jplu/tf-xlm-r-ner-40-lang"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name, from_tf=True)

# Define the NER pipeline using the NER tokenizer
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

def split_text(text, tokenizer, max_length=512):
    """Splits the text into chunks of a maximum length."""
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding='max_length')['input_ids'][0]
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def analyze_text(text):
    # Split text if it's too long
    text_chunks = split_text(text, sentiment_tokenizer)
    
    # Get NER results from the full text using the NER tokenizer
    ner_results = []
    for chunk in text_chunks:
        ner_results.extend(ner_pipeline(chunk))
    
    # Get sentiment results from each chunk using the sentiment tokenizer
    sentiment_scores = np.zeros(3)  # Assuming the model has 3 classes: negative, neutral, positive
    for chunk in text_chunks:
        encoded_input = sentiment_tokenizer(chunk, return_tensors='pt')
        output = sentiment_model(**encoded_input)
        scores = output.logits[0].detach().cpu().numpy()
        scores = softmax(scores)
        sentiment_scores += scores
    
    # Calculate average sentiment score
    average_sentiment_scores = sentiment_scores / len(text_chunks)
    # Determine sentiment label based on the highest average score
    sentiment_label = sentiment_config.id2label[np.argmax(average_sentiment_scores)]
    
    # Extract entities and their labels
    entities = [(result['word'], result['entity']) for result in ner_results]
    
    return entities, sentiment_label

# Read the CSV file
df = pd.read_csv('comments2fastText - Train.csv')

# Process only the first 1000 lines
df = df.head(1000)

# Initialize lists to store results
entities_list = []
sentiment_list = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    text = row['Text']
    entities, sentiment = analyze_text(text)
    entities_list.append(entities)
    sentiment_list.append(sentiment)

# Add the results as new columns to the original DataFrame
df['Entities'] = entities_list
df['Sentiment'] = sentiment_list

# Save the updated DataFrame to a new CSV file
df.to_csv('results.csv', index=False)

