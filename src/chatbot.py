import os
import sys
import logging
import weaviate
import json
import pathlib
from llama_index import GPTVectorStoreIndex

from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings.langchain import LangchainEmbedding
# from langchain.chat_models import ChatOpenAI
from llama_index.langchain_helpers.agents import (LlamaToolkit,
                                                  create_llama_chat_agent,
                                                  IndexToolConfig)
from langchain.callbacks import StdOutCallbackHandler
from llama_index import LLMPredictor, ServiceContext, load_index_from_storage
from src.document_loader import DocumentLoader
from src.langchain_wrapper import DistillGPT
from llama_index.vector_stores import WeaviateVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)
INDEX_DIR = os.environ.get("INDEX_DIR", None)
SCHEMA_DIR = pathlib.Path(__file__).resolve(True).parent.parent / "schemas/schema_old.json"
INDEX_SCHEMA_DIR = pathlib.Path(__file__).resolve(True).parent.parent / "schemas/index_schema.json"

from langchain.embeddings import HuggingFaceEmbeddings



with open(SCHEMA_DIR, 'r') as f:
    SCHEMA_DEF = json.load(f)

with open(INDEX_SCHEMA_DIR, 'r') as f:
    INDEX_SCHEMA = json.load(f)


class Chatbot:
    def __init__(self, index_dir=INDEX_DIR):

        # self.template = template
        # self.f_template = f_template
        self.index_dir = index_dir

        self.llm = DistillGPT()
        self.document_loader = DocumentLoader().clean_data()
        self.wv_client = weaviate.Client(
            url=WEAVIATE_URL,
            additional_headers={"X-OpenAI-Api-Key": OPEN_AI_KEY}
        )
        assert self.wv_client.is_live() and self.wv_client.is_ready()

    def configure_vector_db(self,):
        # print(json.dumps(SCHEMA_DEF, indent=3))
        try:
            self.wv_client.schema.create(SCHEMA_DEF)
        except Exception as e:
            print(e)
        # self.wv_client.schema.create_class(INDEX_SCHEMA)

        product_metadata = self.document_loader.clean_data().get_documents()
        with self.wv_client.batch as batch:
            batch.batch_size = 3
            # Batch import all Questions
            for key, d in product_metadata.items():
                # print(f"importing AmazonProduct: {key+1}")
                self.wv_client.batch.add_data_object(d, "AmazonProduct")

    def set_index(self,):
        product_metadata = self.document_loader.get_documents_from_weaviate()
        # for doc in product_metadata[:2000]:
        #     print(doc)

        self.llm.load_model()
        llm_predictor = LLMPredictor(llm=self.llm)

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        embed_model = LangchainEmbedding(langchain_embedding=hf)

        vector_store = WeaviateVectorStore(weaviate_client=self.wv_client,
                                           class_prefix="AmazonProduct")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                                       embed_model=embed_model)
        cur_index = GPTVectorStoreIndex.from_documents(
            documents=product_metadata,
            service_context=service_context,
            storage_context=storage_context,
        )
        storage_context.persist(persist_dir=str(self.index_dir))
        return storage_context

    def get_index(self,):

        try:
            # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
            # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
            vector_store = WeaviateVectorStore(weaviate_client=self.wv_client, 
                                                class_prefix="AmazonProduct")
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.index_dir)
            return load_index_from_storage(storage_context)
        except Exception as e:
            print(e)
            storage_context = self.set_index()
            return load_index_from_storage(storage_context)

    def set_chatbot(self):
        # self.llm = OpenAI(client=None, temperature=0)
        print(self.llm)
        self.configure_vector_db()
        index = self.get_index()

        query_engine = index.as_query_engine(similarity_top_k=3,)
        tool_config = IndexToolConfig(query_engine=query_engine,
                                      name=f"AMAZON Vector Index",
                                      description="Amazon Products Vector index",
                                      tool_kwargs={"return_direct": True})
        toolkit = LlamaToolkit(index_configs=[tool_config])
        memory = ConversationBufferMemory(memory_key="chat_history")
        return create_llama_chat_agent(toolkit, self.llm, memory=memory, verbose=True)