from prompts.prompt_template import SYSTEM_PROMPTS
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name, temperature, task_name, prompt_name, is_length_constrained):
        self.model_name = model_name
        self.temperature = temperature
        self.task_name = task_name
        if prompt_name in SYSTEM_PROMPTS[task_name]:
            self.system_prompt = SYSTEM_PROMPTS[task_name][prompt_name]
        else:
            self.system_prompt = None
        self.prompt_name = prompt_name
        self.is_length_constrained = is_length_constrained
        print(f"Model: {self.model_name}, \
            Temperature: {self.temperature}, \
            Length Constrained: {self.is_length_constrained}, \
            Prompt: {self.system_prompt}")

    @abstractmethod
    def get_response(self, user_query):
        raise NotImplementedError("The method `get_response` must be implemented.")

    @abstractmethod
    def eval_response(self, user_query, response, knowledge):
        raise NotImplementedError("The method `eval_response` must be implemented.")

    def regenerate_until_valid_length(self, user_query):
        response = self.get_response(user_query)
        res_len = len(response.split())
        count_generation = 1
        while res_len > 250:
            response = self.get_response(user_query)
            res_len = len(response.split())
            count_generation += 1

            print(f"Generation count: {count_generation}")

            if count_generation >= 20:
                print("Warning: Generation count >= 20, stopped")
                break
        print(f"Final response length: {res_len}")
        return response, count_generation
