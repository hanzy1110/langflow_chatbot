from src.chatbot import Chatbot

from fastapi import FastAPI
from pydantic import BaseModel
import tracemalloc
tracemalloc.start()

app = FastAPI()


# Define a request model for the chatbot API
class ChatbotRequest(BaseModel):
    message: str


# Define a response model for the chatbot API
class ChatbotResponse(BaseModel):
    response: str


chatbot_model = Chatbot()
agent_chain = chatbot_model.set_chatbot()


@app.post("/chatbot")
def chatbot_endpoint(request: ChatbotRequest) -> ChatbotResponse:
    message = request.message
    response = agent_chain.run(input=message)
    return ChatbotResponse(response=response)
