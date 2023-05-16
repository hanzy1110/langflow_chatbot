from src.chatbot import Chatbot

from fire import Fire
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv(".env.dev")


def get_mode(mode, chatbot):
    match mode:
        case "vector_db":
            chatbot.configure_vector_db()
        case "index":
            chatbot.set_index()
        case "chat":
            agent_chain = chatbot.set_chatbot()
            while True:
                text_input = input("User: ")
                response = agent_chain.run(input=text_input)
                print(f'Agent: {response}')

def main(mode):
    chatbot = Chatbot()
    get_mode(mode, chatbot)


if __name__ == "__main__":
    Fire(main)
