import csv
import os 
import re 

def extract_sentiment(overallsent):
    match = re.search(r"{overallsent: '(\w+)'}", overallsent)
    if match:
        return match.group(1)
    else:
        return None

def wordToValue(word):
    if word == 'positive':
        return 1
    elif word == 'negative':
        return -1
    elif word == 'neutral':
        return 0
    else:
        return print("jobs done!")
    

def read_sentiments(inputfile, gptflag):
    sentiments = [] 
    with open(inputfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if gptflag:
                sentiment = extract_sentiment(row[6])
                if sentiment == None:
                    sentiment = extract_sentiment(row[5])
            else:
                sentiment = row[6]
            value = wordToValue(sentiment)
            sentiments.append(value)
    return sentiments

def aggregate_data(baseline, model):
    true_positive = sum(1 for base, mod in zip(baseline, model) if base == mod == 1 if base is not None and mod is not None)
    true_neutral = sum(1 for base, mod in zip(baseline, model) if base == mod == 0 if base is not None and mod is not None)
    true_negative = sum(1 for base, mod in zip(baseline, model) if base == mod == -1 if base is not None and mod is not None)

    false_positive = sum(1 for base, mod in zip(baseline, model) if base != mod and mod == 1 if base is not None and mod is not None)
    false_neutral = sum(1 for base, mod in zip(baseline, model) if base != mod and mod == 0 if base is not None and mod is not None)
    false_negative = sum(1 for base, mod in zip(baseline, model) if base != mod and mod == -1 if base is not None and mod is not None)

    total = true_positive + true_neutral + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_neutral + true_negative) / total if total > 0 else 0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    print("True Positive: ", true_positive)
    print("True Neutral: ", true_neutral)
    print("True Negative: ", true_negative)
    print("False Positive: ", false_positive)
    print("False Neutral: ", false_neutral)
    print("False Negative: ", false_negative)
    print("Total: ", total)
    
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)

    return accuracy, precision, recall, f1_score

def main():
    inputfile = "results.csv"
    input2 = 'comments2fastText - Train - GPT3.csv'
    groq = 'comments2fastTextGroq.csv'
    fuckinggroqmodel = read_sentiments(groq, True)
    print(fuckinggroqmodel)
    model = read_sentiments(inputfile, False)
    baseline = read_sentiments(input2, True)
    
    aggregate_data(baseline, model)

        

if __name__ == "__main__":
    main()