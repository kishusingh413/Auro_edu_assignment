import openai
import numpy as np
import os
import time
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Generates an embedding for the given text using OpenAI's API.
def generate_embedding(text, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return np.array(response["data"][0]["embedding"])

        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(delay)
    
    raise RuntimeError("Failed to generate embedding after multiple attempts.")
