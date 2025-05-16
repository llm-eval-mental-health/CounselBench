import numpy as np
import pandas as pd 

df = pd.read_csv('<{MODEL}_EVAlUATED_BY_HUMAN_FILE>', encoding='utf-8')

#Process the data to facilitate subsequent calculations,
df['questionID'] = df['questionID'].str.replace('questionID_', '', regex=False)
final_df_no_model = df.drop(columns=['model'])

# Obtain the data format consistent with the LLM evaluation
final_df_no_model.to_csv("<HUMAN_EVALUATION_FILE_WITH_THE_SAME_FORMAT_AS_THE_LLM_EVALUATION>", index=False)