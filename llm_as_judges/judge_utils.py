import sys
# set path if needed
# sys.path.append('<>')
from models.base_model import BaseModel
from models.gemini import Gemini
from models.openai_llm import OpenAIModel
from models.llama_3 import Llama3Model
from models.claude import ClaudeModel


def model_init(model_name, temperature, task_name, prompt_name, is_length_constrained):

    # abbreviation when generating adversarial examples
    if "gpt" == model_name:
        model_name = "gpt-4-0613"
    elif "llama" == model_name:
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
    elif "gemini" == model_name:
        model_name = "gemini-1.5-pro"
    elif "llama_31" == model_name:
        model_name = "meta-llama/Llama-3.1-70B-Instruct"

    if "gpt" in model_name:
        model = OpenAIModel(model_name=model_name, temperature=temperature, task_name=task_name, prompt_name=prompt_name, is_length_constrained=is_length_constrained)
    elif "llama" in model_name:
        model = Llama3Model(model_name=model_name, temperature=temperature, task_name=task_name, prompt_name=prompt_name, is_length_constrained=is_length_constrained)
    elif "claude" in model_name:
        model = ClaudeModel(model_name=model_name, temperature=temperature, task_name=task_name, prompt_name=prompt_name, is_length_constrained=is_length_constrained)
    elif "gemini" in model_name:
        model = Gemini(model_name=model_name, temperature=temperature, task_name=task_name, prompt_name=prompt_name, is_length_constrained=is_length_constrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model