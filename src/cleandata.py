import csv
import numpy as np
import datetime
import os
from langdetect import detect
import re
import emoji 
import fasttext

fasttext.FastText.eprint = lambda x: None

subreddit_language_dict = {
    "Doubangoosegroup": "Chinese",
    "real_china_irl": "Chinese",
    "liberalgoosegroup": "Chinese",
    "mohu": "Chinese",
    "hanren": "Chinese",
    "Taiwanese": "Chinese",
    "liberta": "Russian",
    "bakchodi": "Indian",
    "indianews": "Indian",
    "Suomi": "Finnish",
    "norge": "Norwegian",
    "de": "German",
    "iceland": "Icelandic",
    "sweden": "Swedish",
    "brasil": "Portuguese",
    "newsokur": "Japanese",
    "hanguk": "Korean",
    "China_irl": "Chinese"

}

fasttext_language_dict = {
    "Chinese": "zh",
    "Russian": "ru",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "English": "en",
    "portuguese": "pt",
    "Icelandic": "is",
    "Swedish": "sv",
    "Norwegian": "no",
    "Finnish": "fi",
    "Indian": "hi",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Dutch": "nl",
    "Polish": "pl",
    "Ukrainian": "uk"
}



def preprocess_text(text, subreddit):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove excessively short messages and normalize text case
    text = text.replace(" ", "")
    if len(text) < 10:
        return None
    return text

def demojify_text(text, subreddit):
    language = subreddit_language_dict[subreddit]
    emojifyLangStr = ""
    match language:
        case "Chinese":
            emojifyLangStr = "zh"
        case "Russian":
            emojifyLangStr = "ru"
        case "German":
            emojifyLangStr = "de"
        case "Japanese":
            emojifyLangStr = "ja"
        case "Korean":
            emojifyLangStr = "ko"
        case _:
            emojifyLangStr = "en"
    text = emoji.demojize(text, language=emojifyLangStr)
    return text


def process_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline= '', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['Author', 'Text', 'Timestamp', 'Subreddit', 'Language']) 
        next(reader)
        for row in reader:
            author = row[0]
            text = row[1]
            timestamp = int(float(row[2]))

            truetime = datetime.date.fromtimestamp(timestamp);
            subreddit = row[3] 
            preprocessed_text = preprocess_text(text, subreddit)
            if preprocessed_text:
                try:
                    language = fastText_detect_language(preprocessed_text, subreddit)
                except:
                    language = "unknown - fasttext error"
                without_emoji = demojify_text(preprocessed_text, subreddit)
                writer.writerow([author, without_emoji, truetime, subreddit, language])


def fastText_detect_language(text, subreddit):
    modelfile = '/home/khaled/NatSec/data/lid.176.bin'
    model = fasttext.load_model(modelfile)
    language = subreddit_language_dict[subreddit]
    lang_code = fasttext_language_dict[language]

    def detect_language(text, k=3):
        predictions = model.predict(text)
        return [(lang.replace('__label__', ''), prob) for lang, prob in zip(predictions[0], predictions[1])]
    language_predictions = detect_language(text)
    weighted_language = max(language_predictions, key=lambda x: x[1] * 1.5 if x[0] == lang_code else x[1])
    return weighted_language[0]

def main():
    inputfile = 'comments2.csv'
    outputfile = 'comments2fastText.csv'
    process_text(inputfile, outputfile)

if __name__ == "__main__":
    main()