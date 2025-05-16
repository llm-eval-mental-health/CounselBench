import google.generativeai as _genai
from google import genai
from google.genai import types
import pdb
import json
from prompts.prompt_template import SYSTEM_PROMPTS
from models.base_model import BaseModel
from prompts.judge_prompts import get_judge_prompt
import time



class Gemini(BaseModel):
    def __init__(self, model_name, temperature, task_name, prompt_name, is_length_constrained):

        super().__init__(model_name, temperature, task_name, prompt_name, is_length_constrained)
    
        # API reference
        # https://ai.google.dev/gemini-api/docs/text-generation?lang=python 
        # https://github.com/google-gemini/generative-ai-python
        GOOGLE_API_KEY = json.load(open("config.json"))["gemini_key"]["<your_name>"]
        if "gemini-2.0" in model_name:
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            _genai.configure(api_key=GOOGLE_API_KEY)
            
    def get_response(self, user_query):

        if "gemini-2.0" in self.model_name:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    max_output_tokens=1024,
                    temperature=self.temperature,
                    candidate_count=1,
                    top_p=1,
                )
            )
            return response.text
        else: # for older models
            # time.sleep(60) # Rate limit for Gemini
            if self.system_prompt == "empty":
                self.model = _genai.GenerativeModel(
                    self.model_name,
                    generation_config=_genai.GenerationConfig(
                    max_output_tokens=1024,
                    top_p=1,
                    candidate_count=1,
                    temperature=self.temperature,)
                )
            else:
                self.model = _genai.GenerativeModel(
                    self.model_name,
                    system_instruction=self.system_prompt,
                    generation_config=_genai.GenerationConfig(
                    max_output_tokens=1024,
                    top_p=1,
                    candidate_count=1,
                    temperature=self.temperature,)
                )
            response = self.model.generate_content(user_query)

        # response = self.model.generate_content(
        #     user_query,
        #     generation_config=genai.types.GenerationConfig(
        #         # Only one candidate for now.
        #         candidate_count=1,
        #         temperature=self.temperature,
        #         top_p=1,
        #         max_output_tokens=1024,
        #     ),
        # )

            return response.text

    def eval_response(self, user_query, response, knowledge):

        prompt = get_judge_prompt(self.prompt_name, user_query, response, knowledge)

        if "gemini-2.0" in self.model_name:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    candidate_count=1,
                    top_p=1,
                    max_output_tokens=1024,
                )
            )
        else: # for older models
            # time.sleep(60) # Rate limit for Gemini
            self.model = _genai.GenerativeModel(
                self.model_name,
                generation_config=_genai.GenerationConfig(
                top_p=1,
                candidate_count=1,
                max_output_tokens=1024,
                temperature=self.temperature,)
            )
            response = self.model.generate_content(prompt)

        return response.text