"""
This file is used to sample 2 responses for each mode from the model outputs for human review. 
"""

import json 
import random
import textwrap
import pandas as pd

def main():
    llama33_response = json.load(open("<llama_33_response_path>"))
    llama31_response = json.load(open("<llama_31_response_path>"))
    gpt35_response = json.load(open("<gpt_35_response_path>"))
    gpt4_response = json.load(open("<gpt_4_response_path>"))
    gemini15_response = json.load(open("<gemini_15_response_path>"))
    gemini2_response = json.load(open("<gemini_2_response_path>"))
    claude35_response = json.load(open("<claude_35_response_path>"))
    claude37_response = json.load(open("<claude_37_response_path>"))

    model_responses = {
        "llama33": llama33_response,
        "llama31": llama31_response,
        "gpt35": gpt35_response,
        "gpt4": gpt4_response,
        "gemini15": gemini15_response,
        "gemini2": gemini2_response,
        "claude35": claude35_response,
        "claude37": claude37_response
    }

    sampled_responses = {}

    for model_name, responses in model_responses.items():
        sampled_responses[model_name] = {}
        for mode in ['apathetic', "assumptions", "symptoms", "judgmental", "medication", "therapy"]: 
            mode_response = []
            new_input_text = set()
            
            for _response in responses:
                if _response["mode"] == mode:
                    mode_response.append(_response)
                    new_input_text.add(_response["new_input_text"])

            assert(len(new_input_text)==20) # 20 unique input text for each mode
            sampled_responses[model_name][mode] = []
            random.seed(42)
            random.shuffle(mode_response)
            first_sampled_question = mode_response.pop(0)
            sampled_responses[model_name][mode].append(first_sampled_question)

            first_question_id = first_sampled_question['question_id']
            first_input_text = first_sampled_question['new_input_text']
            

            find_distinct = False
            while not find_distinct: # keep sampling until we find a distinct question
                random.seed(42)
                random.shuffle(mode_response)
                second_sampled_question = mode_response.pop(0)
                second_question_id = second_sampled_question['question_id']
                second_input_text = second_sampled_question['new_input_text']

                if first_question_id == second_question_id and first_input_text != second_input_text:
                    find_distinct = True
                    sampled_responses[model_name][mode].append(second_sampled_question)
            
            assert(len(sampled_responses[model_name][mode])==2) # 2 sampled responses for each mode

    with open("results/sampled_for_review/sample_for_human_review.json", "w") as f:
        json.dump(sampled_responses, f, indent=4, ensure_ascii=False)

    print("Sampled responses for human review saved to sample_for_human_review.json")


if __name__ == "__main__":
    main()
