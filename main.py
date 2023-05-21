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


# Instantiate your chatbot model here
snap1 = tracemalloc.take_snapshot()
chatbot_model = Chatbot()
agent_chain = chatbot_model.set_chatbot()
snap2 = tracemalloc.take_snapshot()

top_stats = snap2.compare_to(snap1, 'lineno')

print("[ Top 10 differences ]")
for stat in top_stats[:10]:
    print(stat)


@app.post("/chatbot")
def chatbot_endpoint(request: ChatbotRequest) -> ChatbotResponse:
    message = request.message

    # Pass the message to your chatbot model and get the response
    response = agent_chain.run(input=message)

    # Placeholder response for demonstration purposes
    # response = "This is the response from the chatbot model."

    return ChatbotResponse(response=response)
