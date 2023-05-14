import os
import json
import time
import weaviate
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env.dev")

MAX_TEXT_LENGTH=1000  # Maximum num of text characters to use
NUMBER_PRODUCTS = 10  
DATA_PATH = Path(os.getcwd()).resolve() / "data/product_data.csv"
OPEN_AI_KEY = os.environ.get("OPEN_AI_TOKEN", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)
WV_USER = os.environ.get("WV_USER", None)
WV_PASS = os.environ.get("WV_PASS", None)


client = weaviate.Client(
    url = WEAVIATE_URL,  # Replace with your endpoint
    auth_client_secret=weaviate.AuthClientPassword(
        username = WV_USER,  # Replace w/ your WCS username
        password = WV_PASS,  # Replace w/ your WCS password
    ),
    additional_headers = {
        "X-OpenAI-Api-Key": OPEN_AI_KEY  # Replace with your inference API key
    }
)

# ===== add schema =====
# class_obj = {
#     "class": "Question",
#     "vectorizer": "text2vec-openai"
# }
class_obj = {
      "class": "AmazonProduct",
      "description": "Amazon product index items",
      "vectorizer": "text2vec-openai",
      "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
      },
    }
# client.schema.create_class(class_obj)

# ===== import data =====
 

def auto_truncate(val):
    """Truncate the given text."""
    return val[:MAX_TEXT_LENGTH]
 
# Load Product data and truncate long text fields
all_prods_df = pd.read_csv(DATA_PATH, converters={
    'bullet_point': auto_truncate,
    'item_keywords': auto_truncate,
    'item_name': auto_truncate
})

# Replace empty strings with None and drop
all_prods_df['item_keywords'].replace('', None, inplace=True)
all_prods_df.dropna(subset=['item_keywords'], inplace=True)
 
# Reset pandas dataframe index
all_prods_df.reset_index(drop=True, inplace=True)
# Num products to use (subset)
 
# Get the first 2500 products
product_metadata = all_prods_df.head(NUMBER_PRODUCTS).fillna('').to_dict(orient='index')
 
print(list(product_metadata[0].keys()))

with client.batch as batch:
    batch.batch_size=3
    # Batch import all Questions
    for key, d in product_metadata.items():
        print(f"importing AmazonProduct: {key+1}")
        client.batch.add_data_object(d, "AmazonProduct")
        time.sleep(20.0)