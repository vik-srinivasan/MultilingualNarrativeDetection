import csv
import os 
from openai import OpenAI
from config import OPENAI_SECRET
import re
from groq import Groq



def gpt3_baseline(text):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {
        "role": "system",
        "content": "Identify any entities in this data and entity categories (world famous people, locations, organizations, products, ideologies, dates, events) \
                    and the sentiment associated with them. Provide your output in the following format: \
                    {entity: '...', category: '...', sentiment: '...'}\n\
                    Where 'sentiment' is one of 'positive', 'negative', 'category' is the type of entity, and 'entity' is the name of the entity. \
                    in addition, rate the overall sentiment of the text as 'positive', 'negative', or 'neutral'. write it as {overallsent: '...'} \
                    Before providing the format, justify your response. If the entity has a Wikipedia page, use the title of the page as the entity name, otherwise use your best guess of what \
                    name the Wikipedia page would have if it existed. If the entity isn't consequential enough to meet Wikipedia's notability \
                    guidelines, don't include it or include a broader category that it falls under. For my own debugging purposes, after this, please translate the entities"
        },
        {
        "role": "user",
        "content": text
        }
    ],
    temperature=0.2,
    max_tokens=1000,
    top_p=1)
    messages = response.choices[0].message.content

    return messages

def groq_baseline(text):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    model = 'mixtral-8x7b-32768'

    response = client.chat.completions.create(
        messages = [ {
        "role": "system",
        "content": "Identify any entities in this data and entity categories (world famous people, locations, organizations, products, ideologies, dates, events) \
                    and the sentiment associated with them. Provide your output in the following format: \
                    {entity: '...', category: '...', sentiment: '...'}\n\
                    Where 'sentiment' is one of 'positive', 'negative', 'category' is the type of entity, and 'entity' is the name of the entity. \
                    Additionally, provide an overallsentiment label. overallsent: 'positive', 'negative', or 'neutral'. write it as {overallsent: '...'} \n \
                    Do not write anything else."                    
        },
        {
        "role": "user",
        "content": text
        }
        ], 
        model=model,
        temperature=0.2,
        max_tokens=1000,
        top_p=1
    )

    messages = response.choices[0].message.content
    return messages

def process_csv(inputf, outputf):
    with open(inputf, 'r', newline='', encoding='utf-8') as infile, \
        open(outputf, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        headers = next(reader) + ['Entities',  'Sentiments']
        writer.writerow(headers)
        count = 0
        for row in reader:
            count += 1
            if count == 1000:
                break
            text = row[1]
            gptcall = groq_baseline(text)
            overallsentiment, entities = parsegpt3output(gptcall)
            entity_column = ', '.join(entities)
            writer.writerow(row + [entity_column, overallsentiment])



def parsegpt3output(output):
    entity_entries = output.strip().split('\n')
    overall_sentiment = entity_entries.pop()
    entity_entries = entity_entries[:-1]
    return overall_sentiment, entity_entries
    

def main():
    text = '符合我的印象。官二代这类人如果不考虑政治立场比做题家好相处多了。他们通过炫富来维持自己的自尊心，毕竟不学无术也不想被人看不起，通常是靠炫富维持一种优越感来保持自己的价值观不崩坍。如果你跟他们保持距离同时没有看不起他们，他们大多数是愚蠢且坏的光明正大，但是完全不可深交，必须保持距离，一但关系比较近，他们会等级化彼此的关系。做题家往往喜欢当面一套背后一套，同样通过成绩来维持自己的优越感来保持价值观不崩塌。往往都是聪明的两面派，而且都是隐藏起来非常难深交，随时都在猜测你说话背后的意思，交流非常累。 中层的人其实就是普通人，这个群体往往千人千面，好相处的不好相处的都有。 还有一些是OP没接触到的，低端的，走线的，鱼龙混杂，道德底线非常低，但是泥腿子容易交流。 剩下的是有身份，正规移民的，这类人往往是亲属移民，规矩打工，属于不上不下的，可以交朋友，岁静人占比巨大，农村里的华人以此为主'

    inputf = 'comments2fastText - Train.csv'
    outputf = 'comments2fastTextGroq.csv'
    process_csv(inputf, outputf)


        
                
        


if __name__ == "__main__":
    main()
