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

        self.wc_client = weaviate.Client(
            url = WEAVIATE_URL,  # Replace with your endpoint
            additional_headers = {
                "X-OpenAI-Api-Key": OPEN_AI_KEY  # Replace with your inference API key
            }
        )

    def configure_prompt(self, **kwargs):
        self.qa_prompt= PromptTemplate.from_template(self.f_template)
        self.condense_question_prompt = PromptTemplate.from_template(self.template)
        # define two LLM models from OpenAI

    def configure_chain(self):
        
        self.llm = OpenAI(client=None, temperature=0)
        streaming_llm = OpenAI(
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
        question_generator = LLMChain(
            llm=self.llm,
            prompt=self.condense_question_prompt
        )
 
        # use the streaming LLM to create a question answering chain
        doc_chain = load_qa_chain(
            llm=streaming_llm,
            chain_type="stuff",
            prompt=self.qa_prompt
        )

        chatbot = ConversationalRetrievalChain(
            retriever=vectorstore.as_retriever(),
            combine_docs_chain=doc_chain,
            question_generator=question_generator
        )