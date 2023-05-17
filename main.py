from src.chatbot import Chatbot

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Define a request model for the chatbot API
class ChatbotRequest(BaseModel):
    message: str


# Define a response model for the chatbot API
class ChatbotResponse(BaseModel):
    response: str


# Instantiate your chatbot model here
chatbot_model = Chatbot()
agent_chain = chatbot_model.set_chatbot()


# Define the API endpoint for the chatbot
@app.post("/chatbot")
def chatbot_endpoint(request: ChatbotRequest) -> ChatbotResponse:
    message = request.message

    # Pass the message to your chatbot model and get the response
    response = agent_chain.run(input=message)

    # Placeholder response for demonstration purposes
    # response = "This is the response from the chatbot model."

    return ChatbotResponse(response=response)
