import os
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env.dev")

MAX_TEXT_LENGTH = 1000
NUMBER_PRODUCTS = 10
DATA_PATH = Path(os.getcwd()).resolve() / "data/product_data.csv"


def auto_truncate(val):
    """Truncate the given text."""
    return val[:MAX_TEXT_LENGTH]


class DocumentLoader:
    def __init__(self):
        all_prods_df = pd.read_csv(DATA_PATH, converters={
            'bullet_point': auto_truncate,
            'item_keywords': auto_truncate,
            'item_name': auto_truncate
        })

        all_prods_df['item_keywords'].replace('', None, inplace=True)
        all_prods_df.dropna(subset=['item_keywords'], inplace=True)

        all_prods_df.reset_index(drop=True, inplace=True)
        self.product_metadata = all_prods_df.head(
            NUMBER_PRODUCTS).fillna('').to_dict(orient='index')
