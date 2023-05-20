import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from llama_index import WeaviateReader

# from llama_index.readers.weaviate import WeaviateReader
load_dotenv(".env.dev")

MAX_TEXT_LENGTH = 1000
NUMBER_PRODUCTS = 300
DATA_PATH = Path(os.getcwd()).resolve() / "data/product_data.csv"
OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)


def auto_truncate(val):
    """Truncate the given text."""
    return val[:MAX_TEXT_LENGTH]


class DocumentLoader:
    def __init__(self, max_documents=NUMBER_PRODUCTS, data_dir=DATA_PATH, weaviate_url=WEAVIATE_URL):
        self.max_documents = max_documents
        self.data_dir = data_dir
        self.weaviate_url = weaviate_url

        self.all_prods_df = pd.read_csv(self.data_dir, converters={
            'bullet_point': auto_truncate,
            'item_keywords': auto_truncate,
            'item_name': auto_truncate
        })

    def clean_data(self,):
        self.all_prods_df['item_keywords'].replace('', None, inplace=True)
        self.all_prods_df.dropna(subset=['item_keywords'], inplace=True)
        self.all_prods_df.reset_index(drop=True, inplace=True)
        return self

    def get_documents(self,):
        product_metadata = self.all_prods_df.head(
            self.max_documents).fillna('').to_dict(orient='index')

        return product_metadata

    def get_documents_from_weaviate(self, query=None):
        properties = ["marketplace", "country",
                      "bullet_point", "item_keywords", "material",
                      "brand", "color", "item_name",]
        reader = WeaviateReader(self.weaviate_url)
        if query:
            return reader.load_data(class_name="AmazonProduct",
                                    graphql_query=query)
        else:
            return reader.load_data(class_name="AmazonProduct",
                                    properties=properties)
