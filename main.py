from src.chatbot import Chatbot

# create a chat history buffer
chat_history = []
# gather user input for the first question to kick off the bot
question = input("Hi! What are you looking for today?")
 
# keep the bot running in a loop to simulate a conversation
while True:
    result = chatbot(
        {"question": question, "chat_history": chat_history}
    )
    print("\n")
    chat_history.append((result["question"], result["answer"]))
    question = input()
