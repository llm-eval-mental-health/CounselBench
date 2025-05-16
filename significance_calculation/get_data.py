import json
import pandas as pd  

# Load the full human evaluation data from JSON file
with open("<FULL_HUMAN_EVALUTION_FILE_IN_JSON_FORMAT>", "r", encoding="utf-8") as f:
    data = json.load(f)

# Process the scoring data of each question and return the scoring line data of each annotator
def process_data(question_id, component): 
    rows = [] 
    for name, annotator in component.items():
        score_fields = [key for key in annotator if key.endswith("_score")]
        
        rater_ids = annotator.get("survey_id", [])
        num_raters = len(rater_ids)

        for i in range(num_raters):
            row = [question_id, rater_ids[i]]
            for field in score_fields:
                value_list = annotator.get(field, [])
                value = value_list[i] if i < len(value_list) else ""
                row.append(value)
            row.append(name)  
            rows.append(row)
    
    return rows

 # Initialize an empty list to store the rows of all the scoring data
all_rows = []

# Iterate through each question and its corresponding scoring component in the data
for question_id, component in data.items(): 
    rows = process_data(question_id, component)
    all_rows.extend(rows)

# Define the column names for the DataFrame
columns = ["question_id", "rater_id", "overall_score", "empathy_score",
           "specificity_score", "medical_advice_score", "factual_consistency_score", "toxicity_score", "model"]

df = pd.DataFrame(all_rows, columns=columns)

# Save DataFrame to CSV
df.to_csv("<HUMAN_EVALUTION_DATA_FILE>", index=False)

