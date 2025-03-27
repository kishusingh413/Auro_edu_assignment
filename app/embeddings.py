import openai
import numpy as np
import time
import os

from flask import current_app
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate an embedding for the given text using OpenAI's API.
def generate_embedding(text: str, max_retries: int = 3, delay: int = 2) -> np.ndarray:
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return np.array(response["data"][0]["embedding"])
        except Exception as e:
            current_app.logger.error(f"An error occurred: {e}. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(delay)
    raise RuntimeError("Failed to generate embedding after multiple attempts.")