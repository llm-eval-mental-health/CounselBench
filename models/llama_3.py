import torch
import transformers
from prompts.prompt_template import SYSTEM_PROMPTS
from models.base_model import BaseModel
from prompts.judge_prompts import get_judge_prompt

class Llama3Model(BaseModel):
    def __init__(self, model_name, temperature, task_name, prompt_name, is_length_constrained):

        super().__init__(model_name, temperature, task_name, prompt_name, is_length_constrained)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16,
            },
            device_map="auto",
        )

    def get_response(
          self, 
          user_query
      ):
        # API reference: https://github.com/huggingface/huggingface-llama-recipes
        # https://github.com/BunsenFeng/AbstainQA/blob/283c10bb3294d130359937ddd1c695df6782a795/lm_utils.py#L74
        # https://github.com/meta-llama/llama3/issues/42
        if "persona" in self.prompt_name:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query},
            ]
        elif self.prompt_name == "empty":
            messages = []
            messages.append({"role": "user", "content": user_query})
        else:
            raise ValueError("Invalid prompt name")

        outputs = self.pipeline(
                    messages,
                    do_sample=True, 
                    temperature=self.temperature,
                    top_p=1,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id,
                    max_new_tokens=1024 
                )
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def eval_response(self, user_query, response, knowledge):
        prompt = get_judge_prompt(self.prompt_name, user_query, response, knowledge)
        messages = [
                {"role": "user", "content": prompt},
            ]
        if self.temperature == 0:
            outputs = self.pipeline(
                    messages,
                    do_sample=False, # greedy decoding,
                    temperature=None,
                    top_p=1,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id,
                    max_new_tokens=1024
                )
        else:
            outputs = self.pipeline(
                        messages,
                        do_sample=True, 
                        temperature=self.temperature,
                        top_p=1,
                        pad_token_id=self.pipeline.tokenizer.eos_token_id,
                        max_new_tokens=1024
                    )
        response = outputs[0]["generated_text"][-1]['content']
        return response