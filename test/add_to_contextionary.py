import os
import weaviate
from dotenv import load_dotenv
import json

load_dotenv(".env.dev")
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)

SCHEMA_DIR = "schemas/test.json"


with open(SCHEMA_DIR, 'r') as f:
    SCHEMA_DEF = json.load(f)


print(json.dumps(SCHEMA_DEF, indent=3))
wv_client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={"X-OpenAI-Api-Key": OPEN_AI_KEY}
)
# wv_client.contextionary.extend("_", "just an underscore", 1.0)
wv_client.schema.create_class(SCHEMA_DEF)
