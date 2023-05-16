import os
import pandas as pd
from pathlib import Path
 
MAX_TEXT_LENGTH=1000  # Maximum num of text characters to use
NUMBER_PRODUCTS = 2500  
DATA_PATH = Path(os.getcwd()).resolve() / "data/product_data.csv"

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
product_metadata = ( 
    all_prods_df
     .head(NUMBER_PRODUCTS)
     .to_dict(orient='index')
)
 
# Check one of the products
print(product_metadata[0])