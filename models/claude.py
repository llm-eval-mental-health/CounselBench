from anthropic import Anthropic
import pdb
import json
from prompts.prompt_template import SYSTEM_PROMPTS
from models.base_model import BaseModel
from prompts.judge_prompts import get_judge_prompt


class ClaudeModel(BaseModel):
    def __init__(self, model_name, temperature, task_name, prompt_name, is_length_constrained):

        super().__init__(model_name, temperature, task_name, prompt_name, is_length_constrained)
        self.client = Anthropic(
            api_key=json.load(open("config.json"))["claude_key"]["<your_name>"]
        )

    def get_response(self, user_query):

        if "persona" in self.prompt_name:
            messages = [{
                    "role": "user",
                    "content": user_query
                }
            ]
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                system = self.system_prompt, # Role prompt
                temperature=self.temperature,
                top_p=1,
            )
        elif self.prompt_name == "empty":
            messages = [
                {"role": "user", "content": user_query}
            ]
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024, 
                temperature=self.temperature,
                top_p=1,
            )  
        else:
            raise ValueError("Invalid prompt name")

        return response.content[0].text

    def eval_response(self, user_query, response, knowledge): 
        prompt = get_judge_prompt(self.prompt_name, user_query, response, knowledge)

        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024, 
            temperature=self.temperature,
            top_p=1
        ) 
        return response.content[0].text



