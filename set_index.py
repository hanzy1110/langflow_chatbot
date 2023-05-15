import os
from pathlib import Path
from dotenv import load_dotenv

from langchain import OpenAI
from llama_index import GPTListIndex, LLMPredictor, ServiceContext, load_graph_from_storage
from llama_index.indices.composability import ComposableGraph
from llama_index import GPTVectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage

from src.document_loader import DocumentLoader

load_dotenv(".env.dev")

product_metadata = DocumentLoader(100).clean_data().get_documents()
OPEN_AI_KEY = os.environ.get("OPEN_AI_TOKEN", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)


# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))

storage_context = StorageContext.from_defaults()
cur_index = GPTVectorStoreIndex.from_documents(
    documents,
    service_context=llm_predictor,
    storage_context=storage_context,
)
storage_context.persist(persist_dir='/storage/AmazonProducts')


service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
storage_context = StorageContext.from_defaults()

# define a list index over the vector indices
# allows us to synthesize information across each index
# graph = ComposableGraph.from_indices(
#     GPTListIndex,
#     [index_set[y] for y in years], 
#     index_summaries=index_summaries,
#     service_context=service_context,
#     storage_context = storage_context,
# )
# root_id = graph.root_id

# # [optional] save to disk
# storage_context.persist(persist_dir=f'./storage/root')

# # [optional] load from disk, so you don't need to build graph from scratch
# graph = load_graph_from_storage(
#     root_id=root_id, 
#     service_context=service_context,
#     storage_context=storage_context,
# )

