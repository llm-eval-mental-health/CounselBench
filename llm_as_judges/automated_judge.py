import json
import pdb
from tqdm import tqdm
from llm_judge import LLMJudge
import argparse
from scipy.stats import wilcoxon
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from difflib import SequenceMatcher
import statistics
# setting path if needed
# sys.path.append('<>')
from utils import save_metrics, save_predictions
from highlight_text import fig_text
from pathlib import Path

def read_response_by_id(gemini_response_file, gpt_response_file, human_response_file, llama_response_file):
    
    survey_ids = []

    gemini_data = json.load(open(gemini_response_file))
    gpt_data = json.load(open(gpt_response_file))
    human_data = json.load(open(human_response_file))
    llama_data = json.load(open(llama_response_file))

    for question_id in llama_data['questionID']:
        question_idx_llama = llama_data['questionID'].index(question_id)
        assert(llama_data['llm_response'][question_idx_llama] != None) 
        survey_ids.append(f"questionID_{question_id}")

    formatted_data = {
        'input_text': [],
        'gemini_1.5_pro': [],
        'gpt4': [],
        'llama3.3': [],
        'human': [],
        'questionID': []
    }

    for survey_id in survey_ids:
        id_num = int(survey_id.split('_')[-1])
        gemini_idx = gemini_data['questionID'].index(id_num)
        gpt_idx = gpt_data['questionID'].index(id_num)
        llama_idx = llama_data['questionID'].index(id_num)

        human_response = None
        found = False
        for topic in human_data:
            for question in human_data[topic]:
                if question['questionID'] == id_num:
                    human_response = question['answerText']
                    found = True
                    break
            if found:
                break
        assert(human_response is not None)

        human_response = human_response.replace('\xa0',  ' ')

        formatted_data['input_text'].append(gemini_data['input_text'][gemini_idx])
        formatted_data['gemini_1.5_pro'].append(gemini_data['llm_response'][gemini_idx])
        formatted_data['gpt4'].append(gpt_data['llm_response'][gpt_idx])
        formatted_data['llama3.3'].append(llama_data['llm_response'][llama_idx])
        formatted_data['human'].append(human_response)
        formatted_data['questionID'].append(id_num)
    
    return formatted_data    


def calculate_average_score(eval_scores):

    # Exclude "I am not sure" from the average calculation
    average_scores = {key: statistics.mean(eval_scores[key]) if key != "Medical Advice" and key != "Factual Consistency" else eval_scores[key] for key in eval_scores.keys()}
    average_scores['Medical Advice'] = len([x for x in eval_scores['Medical Advice'] if x == "Yes"]) / len([x for x in eval_scores['Medical Advice'] if x != "I am not sure"])
    average_scores['Factual Consistency'] = statistics.mean([x for x in eval_scores['Factual Consistency'] if x != "I am not sure"])
    return average_scores

def is_na(sent):
    return sent.lower().replace(" ", "") in ["nil", "na", "n/a", ""]

def main(args):

    dataset = read_response_by_id(args.gemini_response_file, args.gpt_response_file, args.human_response_file, args.llama_response_file)
    score_category = ["Overall", "Empathy", "Specificity", "Medical Advice", "Factual Consistency", "Toxicity"]
    model_name_mapper = {
        "gemini": "gemini_1.5_pro",
        "gpt4": "gpt4",
        "human": "human",
        "llama3": "llama3.3"
    }
    judge_model = args.judge_model_name
    prompt_strategy = args.prompt_strategy
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    human_annotation_file = args.human_annotation_file

    print("Judge Model:", judge_model)
    print("Prompt Strategy:", prompt_strategy)
    temperature = 0
    judge = LLMJudge(judge_model, temperature, "counsel_chat", prompt_strategy, False, score_category)

    if len(score_category) == 6:
        score_type = "all"
    elif len(score_category) == 1:
        score_type = score_category[0].lower()
    else:
        raise NotImplementedError("Only single score or all scores are implemented for now")

    if prompt_strategy == "classify_overall":
        overall_reasons = json.load(open(human_annotation_file))
        gemini_reasons = []
        for questionID in dataset['questionID']:
            for i, overall_reason in enumerate(overall_reasons[f'questionID_{questionID}']['gemini']["overall_reason"]):
                if overall_reasons[f'questionID_{questionID}']['gemini']["overall_score"][i] <= 2 :
                    gemini_reasons.append({
                        "user_query": dataset['input_text'][dataset['questionID'].index(questionID)],
                        "response": dataset['gemini_1.5_pro'][dataset['questionID'].index(questionID)],
                        "overall_score": overall_reasons[f'questionID_{questionID}']['gemini']["overall_score"][i],
                        "overall_reason": overall_reasons[f'questionID_{questionID}']['gemini']["overall_reason"][i]
                    })
        gemini_outputs = judge.classify_overall_reasons(prompt_strategy, gemini_reasons)
        # save outputs
        save_predictions(gemini_outputs, f"{output_dir}/gemini_overall.json")

        gpt_reasons = []
        for questionID in dataset['questionID']:
            for i, overall_reason in enumerate(overall_reasons[f'questionID_{questionID}']['gpt4']["overall_reason"]):
                if overall_reasons[f'questionID_{questionID}']['gpt4']["overall_score"][i] <= 2 :
                    gpt_reasons.append({
                        "user_query": dataset['input_text'][dataset['questionID'].index(questionID)],
                        "response": dataset['gpt4'][dataset['questionID'].index(questionID)],
                        "overall_score": overall_reasons[f'questionID_{questionID}']['gpt4']["overall_score"][i],
                        "overall_reason": overall_reasons[f'questionID_{questionID}']['gpt4']["overall_reason"][i]
                    })
        gpt_outputs = judge.classify_overall_reasons(prompt_strategy, gpt_reasons)
        # save outputs
        save_predictions(gpt_outputs, f"{output_dir}/gpt4_overall.json")

        human_reasons = []
        for questionID in dataset['questionID']:
            for i, overall_reason in enumerate(overall_reasons[f'questionID_{questionID}']['human']["overall_reason"]):
                if overall_reasons[f'questionID_{questionID}']['human']["overall_score"][i] <= 2 :
                    human_reasons.append({
                        "user_query": dataset['input_text'][dataset['questionID'].index(questionID)],
                        "response": dataset['human'][dataset['questionID'].index(questionID)],
                        "overall_score": overall_reasons[f'questionID_{questionID}']['human']["overall_score"][i],
                        "overall_reason": overall_reasons[f'questionID_{questionID}']['human']["overall_reason"][i]
                    })
        human_outputs = judge.classify_overall_reasons(prompt_strategy, human_reasons)
        # save outputs
        save_predictions(human_outputs, f"{output_dir}/human_overall.json")

        llama_reasons = []
        for questionID in dataset['questionID']:
            for i, overall_reason in enumerate(overall_reasons[f'questionID_{questionID}']['llama3']["overall_reason"]):
                if overall_reasons[f'questionID_{questionID}']['llama3']["overall_score"][i] <= 2 :
                    llama_reasons.append({
                        "user_query": dataset['input_text'][dataset['questionID'].index(questionID)],
                        "response": dataset['llama3.3'][dataset['questionID'].index(questionID)],
                        "overall_score": overall_reasons[f'questionID_{questionID}']['llama3']["overall_score"][i],
                        "overall_reason": overall_reasons[f'questionID_{questionID}']['llama3']["overall_reason"][i]
                    })
        llama3_outputs = judge.classify_overall_reasons(prompt_strategy, llama_reasons)
        # save outputs
        save_predictions(llama3_outputs, f"{output_dir}/llama3_overall.json")

    elif prompt_strategy == "ask_for_toxic_sentence":
        all_data = json.load(open(human_annotation_file))
        toxic_outputs = {}

        for model_name in ['gemini', 'gpt4', 'human', 'llama3']:
            toxic_outputs[model_name] = {}
            count_na = 0
            for questionID in tqdm(dataset['questionID']):
                for i, tox_copy in enumerate(all_data[f'questionID_{questionID}'][model_name]["toxicity_copy"]):
                    if (isinstance(tox_copy, str)) and is_na(tox_copy) == False:
                        toxic_outputs[model_name][f"questionID_{questionID}"] = {
                            "user_query": dataset['input_text'][dataset['questionID'].index(questionID)],
                            "questionID": questionID,
                            "response": dataset[model_name_mapper[model_name]][dataset['questionID'].index(questionID)],
                        }
                        tox_sent = judge.ask_toxic_sentence(prompt_strategy, toxic_outputs[model_name][f"questionID_{questionID}"])
                        if is_na(tox_sent) or "n/a" in tox_sent.lower():
                            count_na += 1
                        toxic_outputs[model_name][f"questionID_{questionID}"]["llm_find_toxicity"] = tox_sent
                        toxic_outputs[model_name][f"questionID_{questionID}"]["toxicity_score"] = []
                        toxic_outputs[model_name][f"questionID_{questionID}"]["toxicity_copy"] = []
                        toxic_outputs[model_name][f"questionID_{questionID}"]["toxicity_reason"] = []
                        toxic_outputs[model_name][f"questionID_{questionID}"]["ro_ratio"] = []
                        break
                for i, tox_copy in enumerate(all_data[f'questionID_{questionID}'][model_name]["toxicity_copy"]):
                    if (isinstance(tox_copy, str)) and is_na(tox_copy) == False:
                        toxic_outputs[model_name][f"questionID_{questionID}"]["toxicity_score"].append(all_data[f'questionID_{questionID}'][model_name]["toxicity_score"][i])
                        toxic_outputs[model_name][f"questionID_{questionID}"]["toxicity_copy"].append(all_data[f'questionID_{questionID}'][model_name]["toxicity_copy"][i])
                        toxic_outputs[model_name][f"questionID_{questionID}"]["toxicity_reason"].append(all_data[f'questionID_{questionID}'][model_name]["toxicity_reason"][i])
                        ro_ratio = SequenceMatcher(None, toxic_outputs[model_name][f"questionID_{questionID}"]["llm_find_toxicity"], all_data[f'questionID_{questionID}'][model_name]["toxicity_copy"][i]).ratio()
                        toxic_outputs[model_name][f"questionID_{questionID}"]["ro_ratio"].append(ro_ratio)

            print(f"[{model_name}]: {count_na} NA found out of {len(toxic_outputs[model_name])} questions")
            Path(f"{output_dir}/toxic_sentences").mkdir(parents=True, exist_ok=True)
            save_predictions(toxic_outputs[model_name], f"{output_dir}/toxic_sentences/{model_name}_response_{prompt_strategy}_by_{judge_model.replace('/', '-')}.json")
            print(f"[Saved {model_name} toxic sentences]") 

    elif prompt_strategy == "ask_for_inconsistent_sentence":
        all_data = json.load(open(human_annotation_file))
        inconsistent_outputs ={}
        for model_name in ['gemini', 'gpt4', 'human', 'llama3']:
            inconsistent_outputs[model_name] = {}
            count_na = 0
            for questionID in tqdm(dataset['questionID']):
                for i, inconsistent_copy in enumerate(all_data[f'questionID_{questionID}'][model_name]["factual_copy"]):
                    if (isinstance(inconsistent_copy, str)) and is_na(inconsistent_copy) == False:
                        inconsistent_outputs[model_name][f"questionID_{questionID}"] = {
                            "user_query": dataset['input_text'][dataset['questionID'].index(questionID)],
                            "questionID": questionID,
                            "response": dataset[model_name_mapper[model_name]][dataset['questionID'].index(questionID)],
                        }
                        inconsistent_sent = judge.ask_inconsistent_sentence(prompt_strategy, inconsistent_outputs[model_name][f"questionID_{questionID}"])
                        if is_na(inconsistent_sent) or "n/a" in inconsistent_sent.lower():
                            count_na += 1
                        inconsistent_outputs[model_name][f"questionID_{questionID}"]["llm_find_inconsistency"] = inconsistent_sent
                        inconsistent_outputs[model_name][f"questionID_{questionID}"]["factual_consistency_score"] = []
                        inconsistent_outputs[model_name][f"questionID_{questionID}"]["factual_copy"] = []
                        inconsistent_outputs[model_name][f"questionID_{questionID}"]["factual_reason"] = []
                        inconsistent_outputs[model_name][f"questionID_{questionID}"]["ro_ratio"] = []
                        break
                for i, inconsistent_copy in enumerate(all_data[f'questionID_{questionID}'][model_name]["factual_copy"]):
                    if (isinstance(inconsistent_copy, str)) and is_na(inconsistent_copy) == False:
                        inconsistent_outputs[model_name][f"questionID_{questionID}"]["factual_consistency_score"].append(all_data[f'questionID_{questionID}'][model_name]["factual_consistency_score"][i])
                        inconsistent_outputs[model_name][f"questionID_{questionID}"]["factual_copy"].append(all_data[f'questionID_{questionID}'][model_name]["factual_copy"][i])
                        inconsistent_outputs[model_name][f"questionID_{questionID}"]["factual_reason"].append(all_data[f'questionID_{questionID}'][model_name]["factual_reason"][i])
                        ro_ratio = SequenceMatcher(None, inconsistent_outputs[model_name][f"questionID_{questionID}"]["llm_find_inconsistency"], all_data[f'questionID_{questionID}'][model_name]["factual_copy"][i]).ratio()
                        inconsistent_outputs[model_name][f"questionID_{questionID}"]["ro_ratio"].append(ro_ratio)

            print(f"[{model_name}]: {count_na} NA found out of {len(inconsistent_outputs[model_name])} questions")
            Path(f"{output_dir}/factual_inconsistency_sentences").mkdir(parents=True, exist_ok=True)
            save_predictions(inconsistent_outputs[model_name], f"{output_dir}/factual_inconsistency_sentences/{model_name}_response_{prompt_strategy}_by_{judge_model.replace('/', '-')}.json")
            print(f"[Saved {model_name} inconsistent sentences]") 

    elif prompt_strategy == "ask_for_medical_advice_sentence":
        all_data = json.load(open(human_annotation_file))
        medical_advice_outputs ={}
        for model_name in ['gemini', 'gpt4', 'human', 'llama3']:
            medical_advice_outputs[model_name] = {}
            count_na = 0
            for questionID in tqdm(dataset['questionID']):
                for i, medical_advice_copy in enumerate(all_data[f'questionID_{questionID}'][model_name]["medical_copy"]):
                    if (isinstance(medical_advice_copy, str)) and is_na(medical_advice_copy) == False:
                        medical_advice_outputs[model_name][f"questionID_{questionID}"] = {
                            "user_query": dataset['input_text'][dataset['questionID'].index(questionID)],
                            "questionID": questionID,
                            "response": dataset[model_name_mapper[model_name]][dataset['questionID'].index(questionID)],
                            "medical_advice_score": [],
                            "medical_copy": [],
                            "medical_reason": [],
                            "ro_ratio": []
                        }
                        medical_advice_sent = judge.ask_medical_advice_sentence(prompt_strategy, medical_advice_outputs[model_name][f"questionID_{questionID}"])
                        if is_na(medical_advice_sent) or "n/a" in medical_advice_sent.lower():
                            count_na += 1
                        medical_advice_outputs[model_name][f"questionID_{questionID}"]["llm_find_medical_advice"] = medical_advice_sent
                        break
                for i, medical_advice_copy in enumerate(all_data[f'questionID_{questionID}'][model_name]["medical_copy"]):
                    if (isinstance(medical_advice_copy, str)) and is_na(medical_advice_copy) == False:
                        medical_advice_outputs[model_name][f"questionID_{questionID}"]["medical_advice_score"].append(all_data[f'questionID_{questionID}'][model_name]["medical_advice_score"][i])
                        medical_advice_outputs[model_name][f"questionID_{questionID}"]["medical_copy"].append(all_data[f'questionID_{questionID}'][model_name]["medical_copy"][i])
                        medical_advice_outputs[model_name][f"questionID_{questionID}"]["medical_reason"].append(all_data[f'questionID_{questionID}'][model_name]["medical_reason"][i])
                        ro_ratio = SequenceMatcher(None, medical_advice_outputs[model_name][f"questionID_{questionID}"]["llm_find_medical_advice"], all_data[f'questionID_{questionID}'][model_name]["medical_copy"][i]).ratio()
                        medical_advice_outputs[model_name][f"questionID_{questionID}"]["ro_ratio"].append(ro_ratio)

            print(f"[{model_name}]: {count_na} NA found out of {len(medical_advice_outputs[model_name])} questions")

            Path(f"{output_dir}/medical_advice_sentences").mkdir(parents=True, exist_ok=True)
            save_predictions(medical_advice_outputs[model_name], f"{output_dir}/medical_advice_sentences/{model_name}_response_{prompt_strategy}_by_{judge_model.replace('/', '-')}.json")
            print(f"[Saved {model_name} medical advice sentences]")  
    elif prompt_strategy == "single_eval": # for scores only
        all_average_scores = {}
        raw_outputs = {}
        all_scores = {}

        for model in ['gpt4', 'llama3.3', 'gemini_1.5_pro', 'human']: 
            eval_scores, raw_eval = judge.evaluate_responses(prompt_strategy, dataset['input_text'], dataset[model], dataset['questionID'], is_cot=False)
            average_scores = calculate_average_score(eval_scores)
            all_average_scores[model] = average_scores
            raw_outputs[model] = raw_eval
            all_scores[model] = eval_scores
            eval_scores['questionID'] = dataset['questionID']
            save_metrics(average_scores, f"{output_dir}/100_{prompt_strategy}_{temperature}_{model}_{score_type}_by_{judge_model.replace('/', '-')}.json")
            save_predictions(eval_scores, f"{output_dir}/100_{prompt_strategy}_{temperature}_{model}_{score_type}_by_{judge_model.replace('/', '-')}.csv")

        save_predictions(raw_eval, f"{output_dir}/100_raw_{prompt_strategy}_{temperature}_{model}_{score_type}_by_{judge_model.replace('/', '-')}.csv")
        save_metrics(all_average_scores, f"{output_dir}/100_{prompt_strategy}_{temperature}_average_{score_type}_by_{judge_model.replace('/', '-')}.json")
    else:
        raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model_name", type=str, required=True, choices=[
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gpt-3.5-turbo",
        "gpt-4.1",
        "gpt-4-0613",])
    parser.add_argument("--prompt_strategy", type=str, required=True, choices=[
        "classify_overall", # classify overall reasons for responses with low overall scores (<=2)
        "ask_for_toxic_sentence",  # ask the judge model to copy and paste toxic sentences
        "ask_for_inconsistent_sentence", # ask the judge model to copy and paste incorrect sentences
        "ask_for_medical_advice_sentence", # ask the judge model to copy and paste medical advice sentences
        "single_eval"]) # ask the judge model to evaluate the response [scores only]
    parser.add_argument("--gemini_response_file", type=str, required=True)
    parser.add_argument("--gpt_response_file", type=str, required=True)
    parser.add_argument("--human_response_file", type=str, required=True)
    parser.add_argument("--llama_response_file", type=str, required=True)
    parser.add_argument("--human_annotation_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)