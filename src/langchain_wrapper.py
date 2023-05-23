from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class DistillGPT(LLM):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model: str = "distilgpt2"
    tokenizer: str = "distilgpt2"
    # pipeline = pipeline("question-answering", model=model, device=device,
    #                     model_kwargs={"torch_dtype":torch.bfloat16})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
        self.model = AutoModelForCausalLM.from_pretrained(
            'distilgpt2').to(self.device)

    def preprocess_input(self, input_text):
        input_ids = self.tokenizer.encode(
            input_text, return_tensors='pt').to(self.device)
        return input_ids

    def generate_response(self, input_ids):
        output = self.model.generate(
            input_ids, max_length=2000, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def _call(self, prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, ) -> str:

        if self.model is None or self.tokenizer is None:
            self.load_model()

        input_ids = self.preprocess_input(prompt)
        response = self.generate_response(input_ids)

        # response = self.pipeline()

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"tokenizer": self.tokenizer,
                "model": self.model,
                "device": self.device}
