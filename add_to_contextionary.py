import os
import weaviate
from dotenv import load_dotenv
load_dotenv(".env.dev")
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)

wv_client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={"X-OpenAI-Api-Key": OPEN_AI_KEY}
)
wv_client.contextionary.extend("_", "just an underscore", 1.0)
