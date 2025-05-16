import numpy as np
import pandas as pd 

# Load the human evaluation data from a CSV file
df = pd.read_csv('<HUMAN_EVALUTION_DATA_FILE>', encoding='utf-8')


# Group the data by 'question_id' and 'model' and calculate the mean for each group 
avg_df = df.groupby(['question_id', 'model'], as_index=False).mean(numeric_only=True)

# Rename the columns for better readability
avg_df_renamed = avg_df.rename(columns={
    'overall_score': 'Overall',
    'empathy_score': 'Empathy',
    'specificity_score': 'Specificity',
    'toxicity_score': 'Toxicity',
    'question_id': 'questionID'
})

final_df = avg_df_renamed[['Overall', 'Empathy', 'Specificity', 'Toxicity', 'questionID', 'model']]

models = final_df['model'].unique()

# Iterate over each model and save its corresponding data to a CSV file
for model in models:
    model_df = final_df[final_df['model']==model]
    model_df.to_csv("<{MODEL}_EVAlUATED_BY_HUMAN_FILE>", index=False)

