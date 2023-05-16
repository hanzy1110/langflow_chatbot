import os
from pathlib import Path
from dotenv import load_dotenv

from langchain import OpenAI
from llama_index import GPTListIndex, LLMPredictor, ServiceContext, load_graph_from_storage
from llama_index.indices.composability import ComposableGraph
from llama_index import GPTVectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage

from src.document_loader import DocumentLoader

load_dotenv(".env.dev")

product_metadata = DocumentLoader(100).get_documents_from_weaviate()
OPEN_AI_KEY = os.environ.get("OPEN_AI_TOKEN", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)
INDEX_DIR = os.environ.get("INDEX_DIR", None)


# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))

storage_context = StorageContext.from_defaults()
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
cur_index = GPTVectorStoreIndex.from_documents(
    product_metadata,
    service_context=service_context,
    storage_context=storage_context,
)
storage_context.persist(persist_dir=str(INDEX_DIR))
