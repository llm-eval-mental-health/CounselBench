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
import nltk
import re
nltk.download('punkt')

def if_contained(a, b):
    # a is shorter than b
    
    PUNCT_RE  = re.compile(r"[^\w\s]")
    SPACE_RE  = re.compile(r"\s+")

    a_norm = SPACE_RE.sub(" ", PUNCT_RE.sub("", a.lower())).strip()
    b_norm = SPACE_RE.sub(" ", PUNCT_RE.sub("", b.lower())).strip()

    # we tried seq_matcher but didn't use it in the end
    seq_matcher = SequenceMatcher(None, a_norm, b_norm)
    match = seq_matcher.find_longest_match(0, len(a_norm), 0, len(b_norm))

    if a_norm == "" or b_norm == "":
        return False

    if a_norm in b_norm: 
        return True

    else:
        return False

def find_distinct_sentences(all_copies, complete_response, all_reasons):

    sub_sentences = nltk.sent_tokenize(complete_response)
    sent_frequency = {}
    
    count_not_found = 0
    
    # edge case when the copy is "all"
    calibrated_copies = []
    calibrated = False
    for i, copy in enumerate(all_copies):
        if copy.lower() in ["all", "all of it", "all of it.", "the entire thing", "all of this.", "most of it.", "entire response"]:
            calibrated_copies.append(complete_response)
            calibrated = True
        elif copy.lower() in ["nothing wrong with it.", "n/.a"]:
            calibrated_copies.append("")
        else:
            calibrated_copies.append(copy)

    for i, copy in enumerate(calibrated_copies):
        sub_copies = nltk.sent_tokenize(copy)

        for sub_copy in sub_copies:
            found_sentence = False
            for sub_sentence in sub_sentences:
                shorter_sentence = sub_copy if len(sub_copy) <= len(sub_sentence) else sub_sentence
                longer_sentence = sub_copy if len(sub_copy) > len(sub_sentence) else sub_sentence
                if if_contained(shorter_sentence, longer_sentence):
                    if sub_sentence not in sent_frequency.keys():
                        sent_frequency[sub_sentence] = {}
                        sent_frequency[sub_sentence]["freq"] = 1
                        sent_frequency[sub_sentence]['reason'] = [all_reasons[i]]
                    else:
                        sent_frequency[sub_sentence]['freq'] += 1
                        sent_frequency[sub_sentence]['reason'].append(all_reasons[i])
                    found_sentence = True

            if not found_sentence: # temprarily ignore the annotation where annotators didn't exactly copy from the response
                count_not_found += 1
                continue
    return sent_frequency, count_not_found

def main(args):
    judge_models = args.model_name
    metric = args.metric

    folder_mapping = {
        "factual_consistency": "factual_inconsistency_sentences",
        "toxicity": "toxic_sentences",
        "medical_advice": "medical_advice_sentences",
    }
    filename_addition_mapping = {
        "factual_consistency": "ask_for_inconsistent_sentence",
        "toxicity": "ask_for_toxic_sentence",
        "medical_advice": "ask_for_medical_advice_sentence",
    }
    copy_mapping = {
        "factual_consistency": "factual_copy",
        "toxicity": "toxicity_copy",
        "medical_advice": "medical_copy",
    }
    reason_mapping = {
        "factual_consistency": "factual_reason",
        "toxicity": "toxicity_reason",
        "medical_advice": "medical_reason",
    }

    metric2llm_key = {
        "factual_consistency": "llm_find_inconsistency",
        "toxicity": "llm_find_toxicity",
        "medical_advice": "llm_find_medical_advice",
    }

    missed_freq_stat = {}

    # for model_name in copied_sentences_from.keys():
    for judge_model in judge_models:

        copied_sentences_from = {}
        changed_judge_model = judge_model
        if judge_model == "meta-llama/Llama-3.3-70B-Instruct":
            changed_judge_model = "llama3.3"
        elif judge_model == "meta-llama/Llama-3.1-70B-Instruct":
            changed_judge_model = judge_model.replace('/', '-')
        copied_sentences_from['gemini'] = json.load(open(f"results/{filename_addition_mapping[metric]}/{judge_model}/{folder_mapping[metric]}/gemini_response_{filename_addition_mapping[metric]}_by_{changed_judge_model}.json"))
        copied_sentences_from['gpt'] = json.load(open(f"results/{filename_addition_mapping[metric]}/{judge_model}/{folder_mapping[metric]}/gpt4_response_{filename_addition_mapping[metric]}_by_{changed_judge_model}.json"))
        copied_sentences_from['human'] = json.load(open(f"results/{filename_addition_mapping[metric]}/{judge_model}/{folder_mapping[metric]}/human_response_{filename_addition_mapping[metric]}_by_{changed_judge_model}.json"))
        copied_sentences_from['llama'] = json.load(open(f"results/{filename_addition_mapping[metric]}/{judge_model}/{folder_mapping[metric]}/llama3_response_{filename_addition_mapping[metric]}_by_{changed_judge_model}.json"))
        distinct_copied_sentences = {}

        stat = {
            "labeled_by_one": {},
            "labeled_by_two": {},
            "not_matched": {}
        }
        missed_freq_stat[judge_model] = {}

        for model_name in ['gpt', "llama", "gemini", "human"]:
        # for model_name in ['human']:
            na_count = 0
            distinct_count = 0
            distinct_copied_sentences[model_name] = {}
            frequent_count = 0
            frequent_miss_count = 0
            total_count_not_found = 0
            for question_id in copied_sentences_from[model_name].keys():
                llm_find_copy = copied_sentences_from[model_name][question_id][metric2llm_key[metric]]
                if 'n/a' in llm_find_copy.lower():
                    na_count += 1 
                all_copies = copied_sentences_from[model_name][question_id][copy_mapping[metric]]
                all_reasons = copied_sentences_from[model_name][question_id][reason_mapping[metric]] 
                response = copied_sentences_from[model_name][question_id]['response']
                distinct_copied_sentences[model_name][question_id], count_not_found = find_distinct_sentences(all_copies, response, all_reasons)
                total_count_not_found += count_not_found
                distinct_count += len(distinct_copied_sentences[model_name][question_id])
                for sentence in distinct_copied_sentences[model_name][question_id].keys():
                    assert(distinct_copied_sentences[model_name][question_id][sentence]['freq'] <= 5)
                    if distinct_copied_sentences[model_name][question_id][sentence]['freq'] > 1:
                        frequent_count += 1
                        found_sentence = False
                        if 'n/a' in llm_find_copy.lower():
                            frequent_miss_count += 1
                        else:
                            llm_single_copies = nltk.sent_tokenize(llm_find_copy)
                            for llm_single_copy in llm_single_copies:
                                shorter_sentence = llm_single_copy if len(llm_single_copy) <= len(sentence) else sentence
                                longer_sentence = llm_single_copy if len(llm_single_copy) > len(sentence) else sentence
                                if if_contained(shorter_sentence, longer_sentence):
                                    found_sentence = True
                            if not found_sentence:
                                frequent_miss_count += 1
                        distinct_copied_sentences[model_name][question_id][sentence]['llm_find_copy'] = llm_find_copy
                        distinct_copied_sentences[model_name][question_id][sentence]['if_llm_match_human_copy'] = found_sentence
                        distinct_copied_sentences[model_name][question_id][sentence]['response'] = response
                        distinct_copied_sentences[model_name][question_id][sentence]['user_query'] = copied_sentences_from[model_name][question_id]['user_query']
                        distinct_copied_sentences[model_name][question_id][sentence]['respondent_model'] = model_name

                        assert(distinct_copied_sentences[model_name][question_id][sentence]['freq'] == len(distinct_copied_sentences[model_name][question_id][sentence]['reason']))
            stat['labeled_by_one'][model_name] = distinct_count
            stat['labeled_by_two'][model_name] = frequent_count
            stat['not_matched'][model_name] = total_count_not_found
            missed_freq_stat[judge_model][model_name] = frequent_miss_count
            save_predictions(distinct_copied_sentences[model_name], f"results/{filename_addition_mapping[metric]}/ignore_{model_name}_distinct_copied_{metric}_by_{changed_judge_model}.json")
            print(f"{model_name} distinct copied sentences saved")
        
        missed_str = ""
        for model_name in ['gpt', "llama", "gemini", "human"]:
            missed_str += "& " + str(missed_freq_stat[judge_model][model_name])
        print(f"Judge {judge_model} Missed Frequencies: {missed_str}")

    count_one_str = ""
    not_matched_str = ""
    count_two_str = ""
    for model_name in ['gpt', "llama", "gemini", "human"]:
        count_one_str += "& " + str(stat['labeled_by_one'][model_name])
        not_matched_str += "& " + str(stat['not_matched'][model_name])
        count_two_str += "& " + str(stat['labeled_by_two'][model_name])
    print(f"Count One: {count_one_str}")
    print(f"Not Matched: {not_matched_str}")
    print(f"Count Two: {count_two_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, nargs='*', choices=[
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gpt-3.5-turbo",
        "gpt-4-0613",
        "gpt-4.1"])
    parser.add_argument("--metric", type=str, required=True, choices=[
        "factual_consistency",
        "toxicity",
        "medical_advice",
    ])
    args = parser.parse_args()
    main(args)