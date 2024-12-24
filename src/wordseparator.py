

def passages_by_tokenlength(text, tokenLength):
    passages = []
    sentence = ''
    for word in text.split():
        if len(sentence) + len(word) <= tokenLength:
            sentence += ' ' + word
        else:
            passages.append(sentence)
            sentence = word
    passages.append(sentence)
    return passages
    

def test_chinese(text):
    length = len(text)
    print(length)

def main():
    text = '要保衛台灣結果不推動全民皆兵制'
    test_chinese(text)

if __name__ == "__main__":

    main()