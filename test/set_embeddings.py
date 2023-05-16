import os
import json
import time
import weaviate
from pathlib import Path

from src.document_loader import DocumentLoader
from dotenv import load_dotenv
load_dotenv(".env.dev")

MAX_TEXT_LENGTH = 1000
NUMBER_PRODUCTS = 10
DATA_PATH = Path(os.getcwd()).resolve() / "data/product_data.csv"
OPEN_AI_KEY = os.environ.get("OPEN_AI_TOKEN", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)
WV_USER = os.environ.get("WV_USER", None)
WV_PASS = os.environ.get("WV_PASS", None)


client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthClientPassword(
        username=WV_USER, password=WV_PASS,
    ),
    additional_headers={
        "X-OpenAI-Api-Key": OPEN_AI_KEY
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

doc_loader = DocumentLoader(100, None)
product_metadata = doc_loader.clean_data().get_documents()

with client.batch as batch:
    batch.batch_size = 3
    # Batch import all Questions
    for key, d in product_metadata.items():
        print(f"importing AmazonProduct: {key+1}")
        client.batch.add_data_object(d, "AmazonProduct")
        time.sleep(20.0)