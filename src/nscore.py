import csv
import ast
from sklearn.metrics import precision_recall_fscore_support

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
    'I-TIME-negative': 24
}

def read_csv(file_path, limit=None):
    texts, tags, original_labels = [], [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            text = row[4]
            labels = ast.literal_eval(row[5])
            processed_labels = [LABEL_IDS.get(label, 0) for label in labels]
            texts.append(text)
            tags.append(processed_labels)
            original_labels.append(labels)
    return texts, tags, original_labels

# Read the first 100 lines
true_texts, true_tags, true_original_labels = read_csv('full_cleaned_data.csv', limit=None)
pred_texts, pred_tags, pred_original_labels = read_csv('d.csv', limit=None)

# Function to calculate F1 score
def calculate_f1(true_tags, pred_tags):
    flat_true = [item for sublist in true_tags for item in sublist]
    flat_pred = [item for sublist in pred_tags for item in sublist]
    precision, recall, f1, _ = precision_recall_fscore_support(flat_true, flat_pred, average='weighted', zero_division=0)
    return precision, recall, f1

# Function to map sentiment
def sentiment_mapping(label):
    if 'negative' in label:
        return -1
    elif 'positive' in label:
        return 1
    elif 'neutral' in label:
        return 0
    else:
        return 0

# Function to calculate sentiment distance
def calculate_sentiment_distance(true_labels, pred_labels):
    distances = []
    for true, pred in zip(true_labels, pred_labels):
        for t, p in zip(true, pred):
            true_sentiment = sentiment_mapping(t)
            pred_sentiment = sentiment_mapping(p)
            distance = 1 - abs(true_sentiment - pred_sentiment) / 2
            distances.append(distance)
    avg_distance = sum(distances) / len(distances)
    return avg_distance

# Calculate F1 scores for entity recognition
precision, recall, f1_score = calculate_f1(true_tags, pred_tags)
print(f'Entity-Level F1 Score: {f1_score}')

# Calculate average sentiment distance
avg_sentiment_distance = calculate_sentiment_distance(true_original_labels, pred_original_labels)
print(f'Average Sentiment Distance: {avg_sentiment_distance}')

# Calculate N-Score
n_score = (f1_score + avg_sentiment_distance) / 2
print(f'N-Score: {n_score}')
