# import openai
# import numpy as np
# import time
# import os

# from flask import current_app
# from dotenv import load_dotenv

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Function to generate an embedding for the given text using OpenAI's API.
# def generate_embedding(text: str, max_retries: int = 3, delay: int = 2) -> np.ndarray:
#     for attempt in range(max_retries):
#         try:
#             response = openai.embeddings.create(
#                 input=text,
#                 model="text-embedding-ada-002"
#             )
#             return np.array(response["data"][0]["embedding"])
#         except Exception as e:
#             current_app.logger.error(f"An error occurred: {e}. Retrying {attempt + 1}/{max_retries}...")
#             time.sleep(delay)
#     raise RuntimeError("Failed to generate embedding after multiple attempts.")

import torch
import numpy as np
import time
import os

from flask import current_app
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_embedding(text: str, max_retries: int = 3, delay: int = 2) -> np.ndarray:
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract token embeddings
            token_embeddings = outputs.last_hidden_state
            
            # Perform mean pooling to get a single vector for the sentence
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_embeddings = token_embeddings * attention_mask.float()
            summed = torch.sum(masked_embeddings, dim=1)
            summed_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embedding = summed / summed_mask
            
            return mean_embedding.squeeze(0).cpu().numpy()
        
        except Exception as e:
            current_app.logger.error(f"An error occurred: {e}. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(delay)
    raise RuntimeError("Failed to generate embedding after multiple attempts.")