from datasets import load_dataset
import argparse
from tqdm import tqdm
import os
from utils import save_metrics, save_predictions
from models.openai_llm import OpenAIModel
from models.llama_3 import Llama3Model
from models.claude import ClaudeModel
from models.gemini import Gemini
from preprocess.counsel_chat import load_counsel_chat, post_process_counsel_chat
import json
import pandas as pd


def read_example_questions(survey_idx, input_file):

    """
    This function read questions from different formats. 
    Usually we will read just all 100 questions, no input_file and survey_idx is needed.
    """

    assert((survey_idx and input_file)==None) # only one can be true

    if input_file: # read from specific file
        print(f"Reading questions from {input_file}")
        data = json.load(open(input_file))
        question_list = []
        for topic in data:
            if isinstance(data[topic], list): # if it's a list
                question_list.extend(data[topic])
            else: # or an object
                question_list.append(data[topic])
        return question_list

    if survey_idx == None: # read all 100 questions 
        data = json.load(open("data/counsel_chat/finalized/finalized_250_filtered_100_questions.json"))
        question_list = []
        for topic in data:
            question_list.extend(data[topic])
        return question_list
    elif survey_idx == 0: 
        data = json.load(open("data/counsel_chat/surveys/surveys.json"))
    else:
        data = json.load(open(f"data/counsel_chat/surveys/individual/survey_{survey_idx}.json"))

    return data

def clean_text(text):
        
    if text is None:
        return ""
    return text

def post_process(response):
    formatted_response = post_process_counsel_chat(response)
    return formatted_response

def generate_predictions(data, model, dataset_config, task_name, is_length_constrained):
        
    predictions = {
        "input_text": [],
        "llm_response": [],
        "word_count": [],
        "generate_counts": []
    }
    for col in data[0]:
        # initialize the list for each column for sub-survey
        predictions[col] = []

    for q_obj in tqdm(data):

        if q_obj["questionText"] == None:
            input_text = clean_text(q_obj['questionTitle'])
        else:
            input_text = clean_text(q_obj["questionTitle"]) + " " + clean_text(q_obj['questionText'])

        if is_length_constrained:
            response, generate_count = model.regenerate_until_valid_length(input_text)
            predictions["generate_counts"].append(generate_count)
        else:
            response = model.get_response(input_text)
            
        predictions["llm_response"].append(response)
        predictions["input_text"].append(input_text)
        predictions["word_count"].append(len(response.split()))
        
        for col in q_obj:
            predictions[col].append(q_obj[col])
        print("Response:", response)

    return predictions

def main(args):

    model_name = args.model_name
    task_name = args.task_name
    prompt_name = args.prompt_name
    eval_split = args.eval_split
    temperature = args.temperature
    is_length_constrained = args.is_length_constrained
    folder_name = "zero_shot"
    file_name_addition = args.file_name_addition
    survey_idx = args.survey_idx
    input_file = args.input_file

    assert(prompt_name == "persona_survey") # we use the persona_survey prompt for all response generation

    data = read_example_questions(survey_idx, input_file)
    dataset_config = json.load(open(f"preprocess/DATASET_CONFIG.json"))[task_name]

    if "gpt" in model_name:
        model = OpenAIModel(model_name=model_name, temperature=temperature, task_name=task_name, prompt_name=prompt_name, is_length_constrained=is_length_constrained)
    elif "llama" in model_name:
        model = Llama3Model(model_name=model_name, temperature=temperature, task_name=task_name, prompt_name=prompt_name, is_length_constrained=is_length_constrained)
    elif "claude" in model_name:
        model = ClaudeModel(model_name=model_name, temperature=temperature, task_name=task_name, prompt_name=prompt_name, is_length_constrained=is_length_constrained)
    elif "gemini" in model_name:
        model = Gemini(model_name=model_name, temperature=temperature, task_name=task_name, prompt_name=prompt_name, is_length_constrained=is_length_constrained)

    predictions = generate_predictions(data, model, dataset_config, task_name, is_length_constrained)
    prediction_file_path = f"results/{task_name}/{folder_name}/{model_name}/{eval_split}/{prompt_name}_{file_name_addition}_temp={temperature}.json"

    if not os.path.exists(f"results/{task_name}/{folder_name}/{model_name}/{eval_split}"):
        os.makedirs(f"results/{task_name}/{folder_name}/{model_name}/{eval_split}")

    save_predictions(predictions, prediction_file_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=[
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gpt-3.5-turbo",
        "gpt-4.5-preview",
        "gpt-4-0613",
        "gpt-4o"])
    parser.add_argument("--task_name", type=str, default="counsel_chat", choices=[
                                                                         "counsel_chat",
                                                                         ])
    parser.add_argument("--prompt_name", type=str, default="basic")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--file_name_addition", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--survey_idx", type=int, default=None)
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--is_length_constrained", action="store_true")
    args = parser.parse_args()

    main(args)