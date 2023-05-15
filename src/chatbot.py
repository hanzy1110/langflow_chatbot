import os
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
from llama_index.readers.weaviate import WeaviateReader
from llama_index.storage.storage_context import StorageContext

from llama_index import (
    GPTVectorStoreIndex, 
    GPTSimpleKeywordTableIndex, 
    GPTListIndex, 
    GPTVectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.vector_stores import WeaviateVectorStore

load_dotenv(".env.dev")

OPEN_AI_KEY = os.environ.get("OPEN_AI_TOKEN", None)
WEAVIATE_URL = os.environ.get("WV_HOST", None)

TEMPLATE = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done.
Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{question}
\"""
Standalone question:"""
 
 
FRIENDLY_TEMPLATE = """You are a friendly, conversational retail shopping assistant. Use the following context including product names, descriptions, and keywords to show the shopper whats available, help find what they want, and answer any questions.
 
It's ok if you don't know the answer.
Context:\"""
 
{context}
\"""
Question:\"
\"""
 
Helpful Answer:"""
 
class Chatbot:
    def __init__(self, template=TEMPLATE, f_template=FRIENDLY_TEMPLATE):

        self.template = template
        self.f_template = f_template

        self.wv_client = weaviate.Client(
            url=WEAVIATE_URL,
            additional_headers={"X-OpenAI-Api-Key": OPEN_AI_KEY}
        )
        assert self.wv_client.is_live() and self.wv_client.is_ready()

    def configure_retriever(self):
        # TODO
        # load Documents and test!
        self.vector_store = WeaviateVectorStore(weaviate_client=self.wv_client, class_prefix='AmazonProducts')
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = GPTVectorStoreIndex.from_documents(nyc_documents, storage_context=storage_context)
    
    def configure_prompt(self, **kwargs):
        self.qa_prompt = PromptTemplate.from_template(self.f_template)
        self.condense_question_prompt = PromptTemplate.from_template(self.template)
        # define two LLM models from OpenAI

    def configure_chain(self):
        
        self.llm = OpenAI(client=None, temperature=0)
        self.streaming_llm = OpenAI(
            client=None,
            streaming=True,
            callback_manager=CallbackManager([
                StreamingStdOutCallbackHandler()
            ]),
            verbose=True,
            max_tokens=150,
            temperature=0.2
        )
        # use the LLM Chain to create a question creation chain
        self.question_generator = LLMChain(
            llm=self.llm,
            prompt=self.condense_question_prompt
        )
 
        # use the streaming LLM to create a question answering chain
        doc_chain = load_qa_chain(
            llm=self.streaming_llm,
            chain_type="stuff",
            prompt=self.qa_prompt
        )

        self.chatbot = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(),
            combine_docs_chain=doc_chain,
            question_generator=self.question_generator
        )