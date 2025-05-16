from openai import OpenAI
import json
import pdb
import sys
import re
from prompts.prompt_template import SYSTEM_PROMPTS
from models.base_model import BaseModel
from prompts.judge_prompts import get_judge_prompt


class OpenAIModel(BaseModel):

    def __init__(self, model_name, temperature, task_name, prompt_name, is_length_constrained):
        super().__init__(model_name, temperature, task_name, prompt_name, is_length_constrained)
        OPENAI_KEY = json.load(open("config.json"))["openai_key"]["<your_name>"]
        self.chat = OpenAI(api_key=OPENAI_KEY)

    def get_response(self, user_query, json_mode=False, max_tokens=1024):
            
        if  "persona" in self.prompt_name:
            completion = self.chat.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": user_query
                    }
                ],
                temperature=self.temperature,
                max_completion_tokens=1024
                # default top_p = 1
            )
            return completion.choices[0].message.content
        elif self.prompt_name == "empty":
            msg_wo_system_prompt = [
                {"role": "user", "content": user_query}
            ]
            if json_mode:
                wo_completion = self.chat.chat.completions.create(
                    model=self.model_name,
                    messages=msg_wo_system_prompt,
                    temperature=self.temperature,
                    max_completion_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    # default top_p = 1
                )
            else:
                wo_completion = self.chat.chat.completions.create(
                    model=self.model_name,
                    messages=msg_wo_system_prompt,
                    temperature=self.temperature,
                    max_completion_tokens=max_tokens,
                    # default top_p = 1
                )
            return wo_completion.choices[0].message.content
        else:
            raise ValueError("Invalid prompt name")

    def eval_response(self, user_query, response, knowledge): 
        prompt = get_judge_prompt(self.prompt_name, user_query, response, knowledge)
        msg_wo_system_prompt = [
            {"role": "user", "content": prompt}
        ]
        wo_completion = self.chat.chat.completions.create(
            model=self.model_name,
            messages=msg_wo_system_prompt,
            temperature=self.temperature,
            max_completion_tokens=1024,
            # default top_p = 1
        )
        return wo_completion.choices[0].message.content