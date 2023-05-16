import os
import time
import weaviate
from dotenv import load_dotenv

from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate

from langchain.schema import Document
from llama_index.storage.storage_context import StorageContext

from llama_index import (
    GPTVectorStoreIndex, 
    GPTSimpleKeywordTableIndex, 
)

# do imports
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from llama_index import GPTListIndex, LLMPredictor, ServiceContext, load_index_from_storage

from src.document_loader import DocumentLoader
from llama_index.vector_stores import WeaviateVectorStore

load_dotenv("../.env.dev")

OPEN_AI_KEY = os.environ.get("OPEN_AI_TOKEN", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)
INDEX_DIR = os.environ.get("INDEX_DIR", None)

class Chatbot:
    def __init__(self, index_dir=INDEX_DIR):

        # self.template = template
        # self.f_template = f_template
        self.index_dir = index_dir

        self.document_loader = DocumentLoader().clean_data()
        self.wv_client = weaviate.Client(
            url=WEAVIATE_URL,
            additional_headers={"X-OpenAI-Api-Key": OPEN_AI_KEY}
        )
        assert self.wv_client.is_live() and self.wv_client.is_ready()

    def configure_vector_db(self,):
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
        product_metadata = self.document_loader.clean_data().get_documents()
        with self.wv_client.batch as batch:
            batch.batch_size = 3
            # Batch import all Questions
            for key, d in product_metadata.items():
                print(f"importing AmazonProduct: {key+1}")
                self.wv_client.batch.add_data_object(d, "AmazonProduct")
                time.sleep(20.0)

    def set_index(self,):
        product_metadata = self.document_loader.get_documents_from_weaviate()
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))

        storage_context = StorageContext.from_defaults()
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
        cur_index = GPTVectorStoreIndex.from_documents(
            product_metadata,
            service_context=service_context,
            storage_context=storage_context,
        )
        storage_context.persist(persist_dir=str(self.index_dir))

    def get_index(self,):
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
        storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
        return load_index_from_storage(storage_context)
    
    def set_chatbot(self):
        self.llm = OpenAI(client=None, temperature=0)
        index = self.get_index()

        query_engine = index.as_query_engine(similarity_top_k=3,)
        tool_config = IndexToolConfig(query_engine=query_engine, 
                                      name=f"AMAZON Vector Index",
                                      description="Amazon Products Vector index",
                                      tool_kwargs={"return_direct": True})
        toolkit = LlamaToolkit(index_configs=[tool_config])
        memory = ConversationBufferMemory(memory_key="chat_history")
        return create_llama_chat_agent(toolkit, self.llm, memory=memory, verbose=True)