"""
This script is used to clean the qualtrics data (from a csv) and map the questions to issues.
"""

import json
import pandas as pd
import argparse

def map_question_to_issues(input_df, output_path):

    issue2survey_questions = {
        "apathetic": ["Q1.3", "Q1.4"],
        "assumptions": ["Q232", "Q233"],
        "symptoms": ["Q237", "Q238"],
        "judgmental": ["Q242", "Q243"],
        "medication": ["Q247", "Q248"],
        "therapy": ["Q252", "Q253"],
    }

    new_questions = {issue: [] for issue in issue2survey_questions}
    new_questions['annotator_id'] = []

    for idx, row in input_df.iloc[2:].iterrows(): # the first two rows are headers
        for issue, questions in issue2survey_questions.items():
            for question in questions:
                new_questions[issue].append(row[question])
        new_questions['annotator_id'].extend([idx-2] * 2) # since each annotator write two questions per issue

    # save the new questions to csv
    new_questions_df = pd.DataFrame(new_questions)
    new_questions_df.to_csv(output_path, index=False)

def main(args):
    input_df = pd.read_csv(args.input_file)
    cleaned_new_questions = map_question_to_issues(input_df, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run new queries on a model.")
    parser.add_argument("--input_file", type=str, required=True, help="CSV file to read.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the cleaned new questions.")
    args = parser.parse_args()
    main(args)