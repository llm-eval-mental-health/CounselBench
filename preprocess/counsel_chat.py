"""
The file is the process how we filter and select the questions and responses from the Counsel Chat dataset.
"""


from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pdb
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
import json


def post_process_counsel_chat(prediction):
    return prediction # nothing needed to be done, okay this is a little redundant

def hf_data_sanity_check(df):
    ## Check if one quesiton ID have duplicates answers from the same therapist
    ## No duplicates found
    count = 0
    for question_id in df['questionID'].unique():
        question_df = df[df["questionID"] == question_id]
        if (question_df['therapistURL'].nunique() != question_df.shape[0]):
            count +=1
    assert(count == 0)

    # Check if each question has a unique questionID
    for questionTitle in df['questionTitle'].unique():
        if df[df['questionTitle'] == questionTitle]['questionID'].nunique() != 1:  
            # check if two posts with the same question title have different question text
            if df[df['questionTitle'] == questionTitle]['questionID'].nunique() != df[df['questionTitle'] == questionTitle]['questionText'].nunique():
                raise valueError("Found Duplicate Question")
        if df[df['questionTitle'] == questionTitle]['topic'].nunique() != 1:
            if df[df['questionTitle'] == questionTitle]['topic'].nunique() != df[df['questionTitle'] == questionTitle]['questionText'].nunique():
                raise valueError("Found Duplicate Question")

    old_data = pd.read_csv("data/counsel_chat/20200325_counsel_chat.csv")

    # check if the data include all old data
    diff_titles = old_data[~(old_data["questionTitle"].isin(df["questionTitle"]))].dropna()

def filter_length(x):
    if x is None: # if the response is None, return False
        return False
    return len(x.split()) <= 250

def filter_hf_counsel_chat_questions():
    ## Warining: some posts may only have a question title and no question text, so don't simply drop NA values
    ds = load_dataset("nbertagnolli/counsel-chat")
    data_df = ds['train'].to_pandas()

    hf_data_sanity_check(data_df)

    ###Find top 20 topics###
    deduplicated_df = pd.DataFrame()
    for questionID in data_df["questionID"].unique():
        question_df = data_df[data_df["questionID"] == questionID]
        max_upvote = question_df['upvotes'].max() # get the max upvotes
        # if multiple reponses with the same upvotes, sample randomly
        # We choose the most-upvoted response here simply for deduplication
        # this is not the final response
        deduplicated_df = pd.concat([deduplicated_df, question_df[question_df["upvotes"] == max_upvote].sample(1, random_state=42)])
    
    # find topics with the most questions
    value_counts = deduplicated_df["topic"].value_counts()
    # values are sorted in descending order by default
    topics = value_counts.index.tolist()
    topics.remove("intimacy")
    topics.remove('human-sexuality')
    topics.remove("lgbtq")
    topics.remove("spirituality")
    topics = topics[:20]

    ###Filter out questions with too long responses###
    filtered_df = pd.DataFrame()
    for questionID in deduplicated_df["questionID"].unique():
        question_df = data_df[data_df["questionID"] == questionID]
        question_df = question_df[question_df["answerText"].apply(lambda x: filter_length(x))] # filter out all responses > 250 words
        if question_df.shape[0] != 0: # only include questions with valid responses
            max_upvote = question_df['upvotes'].max()
            filtered_df = pd.concat([filtered_df, question_df[question_df["upvotes"] == max_upvote].sample(1, random_state=42)]) # if multiple reponses with the same upvotes, sample randomly

    ###For each topic, choose 5 questions###
    question_dict = {}
    for topic in topics:
        if len(filtered_df[filtered_df["topic"] == topic]) == 5:
            question_dict[topic] = filtered_df[filtered_df["topic"] == topic].iloc[0]["questionText"]
        
        # Print Top5 upvoted questions with the highest upvotes
        sorted_df = filtered_df[filtered_df["topic"] == topic].sort_values("upvotes", ascending=False)

        # For 250 Words: those questions are directly related to religious or sexual content, so we manually delete them 
        if topic == "relationships":
            sorted_df = sorted_df[sorted_df["questionID"] != 766]
            sorted_df = sorted_df[sorted_df["questionID"] != 737]
        if topic == 'self-esteem':
            sorted_df = sorted_df[sorted_df["questionID"] != 351]
            sorted_df = sorted_df[sorted_df["questionID"] != 347]
        if topic == 'counseling-fundamentals':
            sorted_df = sorted_df[sorted_df["questionID"] != 934]

        ### 04/09: Due to the extreme long word count for the post text, and LLama failed to generate a response within 250 words, we later removed the questionID=446
        ### questionID=441 is also filtered out because the response contain an invalid web pdf link, which we are not sure if it used to be valid at the time of responding
        if topic == 'domestic-violence':
            sorted_df = sorted_df[sorted_df["questionID"] != 446]
            sorted_df = sorted_df[sorted_df["questionID"] != 441]

        if len(sorted_df) < 5:
            raise ValueError("Not enough questions for topic")

        count = 0
        top5_upvote_rows = pd.DataFrame()
        sorted_upvotes = sorted(sorted_df['upvotes'].unique(), reverse=True)

        for upvotes in sorted_upvotes: # starting from upvote.max()
            if count == 5:
                break
            if sorted_df[sorted_df["upvotes"] == upvotes].shape[0] + count <= 5:
                # take all questions with the same upvotes
                count += sorted_df[sorted_df["upvotes"] == upvotes].shape[0]
                top5_upvote_rows = pd.concat([top5_upvote_rows, sorted_df[sorted_df["upvotes"] == upvotes]], axis=0)
            else:
                # if there are multiple questions with the same upvotes, sample randomly
                top5_upvote_rows = pd.concat([top5_upvote_rows, sorted_df[sorted_df["upvotes"] == upvotes].sample(n=5 - count, random_state=42)], axis=0)
                break

        top5_upvote_questions = []
        for index, row in top5_upvote_rows.iterrows():
            question_obj = {}
            question_obj['questionTitle'] = row["questionTitle"]
            question_obj['questionText'] = row["questionText"]
            question_obj['questionID']= row['questionID']
            top5_upvote_questions.append(question_obj)
            
        question_dict[topic] = top5_upvote_questions

    with open("data/counsel_chat/finalized/finalized_250_filtered_100_questions.json", "w+") as f:
        json.dump(question_dict, f, ensure_ascii=False, indent=4)

    return question_dict

def get_human_responses(hf_filtered_questions):

    qas_dict = {}

    ds = load_dataset("nbertagnolli/counsel-chat")

    data_df = ds['train'].to_pandas()
    count_q = 0
    num_filtered_best = 0

    for topic in hf_filtered_questions.keys():
        assert(len(hf_filtered_questions[topic]) == 5)

        question_idx_per_row = 2

        qas_dict[topic] = []

        for question_obj in hf_filtered_questions[topic]:
            
            question_id = question_obj["questionID"]
            question_df = data_df[data_df["questionID"] == question_id]
            qa_dict = {}
            qa_dict["questionID"] = question_id
            for column in ['questionTitle', 'questionText', 'questionLink', 'topic']:
                qa_dict[column] = question_df.iloc[0][column] # get question info

            # sanity check 
            assert(question_df['questionTitle'].nunique() == 1) # Every question has a unique question title
            assert(question_df['questionTitle'].unique()[0] == question_obj["questionTitle"]) # Mapped Question Title
            
            count_q += 1
            question_idx_per_row +=1 

            # filter out all responses > 250 words
            prev_len = question_df.shape[0]
            prev_df = question_df

            question_df = question_df[question_df["answerText"].apply(lambda x: filter_length(x))]
            after_len = question_df.shape[0]

            # warning if the most-upvoted responses is filtered out
            # just for statistical purpose
            if prev_df['upvotes'].max() != question_df['upvotes'].max():
                num_filtered_best += 1

            # get the most upvoted response 
            max_upvotes = question_df.sort_values("upvotes", ascending=False).iloc[0]['upvotes']
            most_upvoted_responses = question_df[question_df["upvotes"] == max_upvotes]

            chosen_response = None

            # Choose the most-voted response; otherwise, sample randomly
            if most_upvoted_responses.shape[0] > 1:
                chosen_response = most_upvoted_responses.sample(1, random_state=42).iloc[0]
                qa_dict['seed'] = 42
            else:
                chosen_response = most_upvoted_responses.iloc[0]

            # For 250 Words: those responses may contain religious content or sexual content, so we manually delete them
            # Since they are all most upvoted responses, we can just choose the second most upvoted response
            if question_id in [250, 355]:
                assert(most_upvoted_responses.shape[0] == 1)
                second_upvote = sorted(question_df['upvotes'].unique(), reverse=True)[1] # get the second most upvoted response
                chosen_response = question_df[question_df['upvotes'] == second_upvote].sample(1, random_state=42).iloc[0]
        

            qa_dict.update({"answerText": chosen_response["answerText"], 
                            "therapistURL": chosen_response["therapistURL"], 
                            "therapistInfo": chosen_response["therapistInfo"],
                            "upvotes": int(chosen_response["upvotes"]),
                            "questionIDX": count_q,
                            "views": int(chosen_response["views"])})
            qas_dict[topic].append(qa_dict)

        print('[sanity check passed: ', topic, ']')

    print('[human responses saved]')
    return qas_dict


def main():

    ## Get Questions 
    hf_filtered_questions = filter_hf_counsel_chat_questions()

    ## Get Responses
    human_responses = get_human_responses(hf_filtered_questions)

if __name__ == "__main__":
    main()