from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

MAX_TEXT_OUTPUT = int(os.environ.get("MAX_TEXT_OUTPUT"))


class DistillGPT(LLM):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    # pipeline = pipeline("question-answering", model=model, device=device,
    #                     model_kwargs={"torch_dtype":torch.bfloat16})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def chat_step(self):
        return self.chat_step

    @chat_step.setter
    def chat_step(self, step):
        return self.chat_step + step

    @property
    def chat_history_ids(self):
        return self.chat_history_ids
    
    @chat_history_ids.setter
    def chat_history_ids(self,chat_history_ids_new):
        return chat_history_ids_new

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium").to(self.device)

    def preprocess_input(self, input_text, chat_history_ids, step=0):
        # input_text = "How are you?"
        new_user_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        self.chat_history_ids = bot_input_ids

        self.tokenizer.pad_token = self.tokenizer.eos_token
        # inputs = self.tokenizer.encode_plus(input_text, return_tensors="pt", 
        #                                   padding="longest",
        #                                   truncation=True)
                                          
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        input_ids[input_ids == self.tokenizer.pad_token_id] = self.tokenizer.eos_token_id
        return input_ids, attention_mask, bot_input_ids

    def generate_response(self, input_ids, attention_mask, bot_input_ids):
        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        # pretty print last ouput tokens from bot
        response = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
                                        skip_special_tokens=True,
                                        attention_mask=attention_mask)
        # output = self.model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     # max_length=MAX_TEXT_OUTPUT,
        #     # num_beams=4, 
        #     # early_stopping=True
        # )
        # response = self.tokenizer.decode(output[0], 
                                        # skip_special_tokens=True)
        return response

    def _call(self, prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, ) -> str:

        # if self.model is None or self.tokenizer is None:
        chat_history_ids = []
        self.load_model()
        input_ids, attention_mask, bot_input_ids = self.preprocess_input(prompt, chat_history_ids, self.chat_step)
        chat_history_ids = self.generate_response(input_ids, attention_mask, bot_input_ids)
        # print("-x-"*30)
        # print("RESPONSE FROM MODEL...")
        # print(response)
        # print("-x-"*30)
        self.chat_step += 1
        return chat_history_ids

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"tokenizer": self.tokenizer,
                "model": self.model,
                "device": self.device}
