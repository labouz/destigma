import pandas as pd
import numpy as np
import openai
import json

prompt = """
Please play the role of an emotion recognition expert. Pleae provide the most likely emotion that the following text conveys.
Only one emotion should be provided.

"""
def get_emotion(text, client, model="gpt-4-turbo-2024-04-09", temperature=0, retries = 2):
    while retries > 0:
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,
                model=model
            )
            label = response.choices[0].message.content.lower().strip()
            return label
        except Exception as e:
            retries -= 1
            print(f"Error: {e}")
            print(f"Retries left: {retries}")
            print("Retrying...")
            return "skipped"
    