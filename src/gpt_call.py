import openai
import pandas as pd
import time
import asyncio
import aiohttp

# Set your OpenAI API key
openai.api_key = 'KEY'


def get_entities_and_sentiments(text):
    prompt = f"""
    You are tasked as an advanced NLP model with the capability to identify named entities within a text and assess the sentiment (positive, negative, or neutral) linked to each entity. Your focus should be on entities categorized as PERSON (PER), LOCATION (LOC), ORGANIZATION (ORG), and TIME (TIME).

    For every text input provided, your output should adhere to the structured format below:

    [
      {{
        "entity": "EntityName1",
        "entity_type": "EntityType1",
        "sentiment": "Sentiment1"
      }},
      {{
        "entity": "EntityName2",
        "entity_type": "EntityType2",
        "sentiment": "Sentiment2"
      }},
      ...
    ]

    Example:
    Text Input:
    "Donald Trump was a terrible president in 2018 but Barack Obama was an amazing president in 2012 when he was in Washington DC."

    Expected Output:

    [
      {{
        "entity": "Donald Trump",
        "entity_type": "PER",
        "sentiment": "negative"
      }},
      {{
        "2018",
        "entity_type": "TIME",
        "sentiment": "neutral"
      }},
      {{
        "Barack Obama",
        "entity_type": "PER",
        "sentiment": "positive"
      }},
      {{
        "2012",
        "entity_type": "TIME",
        "sentiment": "neutral"
      }},
      {{
        "Washington DC",
        "entity_type": "LOC",
        "sentiment": "neutral"
      }}
    ]

    Please ensure that your analysis is precise and that each entity is correctly identified and classified according to the 4 categories [PER, ORG, LOC, TIME] and associated sentiment.

    Now please process the following text post:
    **Text Post:** "{text}"
    """

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1000,
        top_p=1
    )

    messages = response.choices[0].message.content

    return messages



def process_csv(input_file, output_file_template, start_row):
    batch_index = start_row // 250  # The index for the output file number
    output_base_name = output_file_template  # The output filename without the index

    reader = pd.read_csv(input_file, chunksize=250, iterator=True, skiprows=range(1, start_row))

    for chunk in reader:
        chunk.columns = [col.lower() for col in chunk.columns]
        
        if 'gpt_output' not in chunk.columns:
            chunk['gpt_output'] = ""

        for row_index, row in chunk.iterrows():
            text_post = row['text']
            gpt_output = get_entities_and_sentiments(text_post)
            chunk.at[row_index, 'gpt_output'] = gpt_output

        output_filename = f"{output_base_name}_{batch_index:03d}.csv"  # Append batch index to filename
        batch_index += 1  # Increment the batch index for the next batch

        chunk.to_csv(output_filename, index=False)

def main():
    input_file = 'comments3fastText.csv'
    output_file = 'comments3finishingViksTrain'
    process_csv(input_file, output_file, start_row=20000)

if __name__ == "__main__":
    main()