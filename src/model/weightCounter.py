

import csv
import ast
from collections import Counter

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


def processWeights(fileinput):
    label_counts = Counter()
    with open(fileinput, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            label_str = row[5]
            labels = ast.literal_eval(label_str)
            entities = [label.split('-')[0] + '-' + label.split('-')[1] if '-' in label else 'O' for label in labels]
            valid_entities = [entity for entity in entities if entity in NER_LABEL_IDS]
            label_counts.update(valid_entities)
    return label_counts

def calculateWeights(label_counts, smooth_factor=1.0):

    total = sum(label_counts.values())
    # Calculate weights as the inverse of frequency
    weights = {label: (total / (count + smooth_factor)) for label, count in label_counts.items()}
    # Normalize weights such that the smallest weight is 1.0
    min_weight = min(weights.values())
    normalized_weights = {label: weight / min_weight for label, weight in weights.items()}
    return normalized_weights
            

def main():
    label_counts = processWeights('cleaned_data.csv')
    weights = calculateWeights(label_counts)
    print(weights)

if __name__ == '__main__':
    main()


    