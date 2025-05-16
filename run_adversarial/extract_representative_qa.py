"""
We identify 6 representative question-answer pairs.
"""

import json

def main():
    data_dict = {}
    data_dict['gpt'] = json.load(open("<gpt_response_path>"))
    data_dict['llama'] = json.load(open("<llama_response_path>"))
    data_dict['gemini'] = json.load(open("<gemini_response_path>"))

    # Example question IDs for each issue
    question_list = {
        "gpt": {
            "medication": 859,
            "therapy": 921
        },
        "llama": {
            "symptoms": 727,
            "judgmental": 346 
        },
        "gemini": {
            "apathetic": 261,
            "assumptions": 76
        }
    }

    examples = {}

    for model_name in data_dict.keys():
        for mode in question_list[model_name].keys():
            question_id = question_list[model_name][mode]
            question_idx = data_dict[model_name]['questionID'].index(question_id)
            input_text = data_dict[model_name]['input_text'][question_idx]
            question_title = data_dict[model_name]['questionTitle'][question_idx]
            question_text = data_dict[model_name]['questionText'][question_idx]
            response_text = data_dict[model_name]['llm_response'][question_idx]
            
            entry = {
                "model_name": model_name,
                "mode": mode,
                "question_id": question_id,
                "input_text": input_text,
                "question_title": question_title,
                "question_text": question_text,
                "response_text": response_text,
            }

            examples[mode] = entry

    output_path = f"data/new_questions/finalized_issues.json"
    
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()