import pandas as pd
import numpy as np
from agreement.utils.transform import pivot_table_frequency
from agreement.utils.kernels import identity_kernel, linear_kernel, ordinal_kernel
from agreement.metrics import cohens_kappa, krippendorffs_alpha, _compute_observed_agreement
from agreement.utils.transform import pivot_table_frequency

# Load human evaluation data
df = pd.read_csv('<HUMAN_EVALUTION_DATA_FILE>', encoding='utf-8')


# Define the columns related to scoring metrics
score_columns = [
    "overall_score", "empathy_score", "specificity_score",
    "factual_consistency_score", "toxicity_score"
]
# Get unique question IDs
question_ids = df["question_id"].unique()

results = []

for qid in question_ids:
    df_q = df[df["question_id"] == qid]
    for score_col in score_columns:
        #The factual_consistency_score includes the option of "I am not sure" and needs to be processed and calculated separately
        if score_col != "factual_consistency_score":
            #Convert the ratings into frequencies
            questions_answers_table =pivot_table_frequency(df_q["model"], df_q[score_col],np.array([1, 2, 3, 4, 5]))
            unweighted_krippendorffs_alpha = krippendorffs_alpha(questions_answers_table)
            linear_weighted_krippendorffs_alpha = krippendorffs_alpha(questions_answers_table, weights_kernel=linear_kernel)
            ordinal_weighted_krippendorffs_alpha = krippendorffs_alpha(questions_answers_table,weights_kernel=ordinal_kernel)
      
            results.append({
                "question_id": qid,
                "score_type": score_col,
                "krippendorffs_alpha_nominal": unweighted_krippendorffs_alpha,
                "krippendorffs_alpha_linear": linear_weighted_krippendorffs_alpha,
                "krippendorffs_alpha_ordinal": ordinal_weighted_krippendorffs_alpha
            })
        else:
            df_q = df_q.copy()
            df_q[score_col] = df_q[score_col].replace("I am not sure", 5).astype(float)
            questions_answers_table =pivot_table_frequency(df_q["model"], df_q[score_col],np.array([1, 2, 3, 4,5]))
            questions_answers_table = questions_answers_table[:, :-1]
            unweighted_krippendorffs_alpha = krippendorffs_alpha(questions_answers_table)
            linear_weighted_krippendorffs_alpha = krippendorffs_alpha(questions_answers_table, weights_kernel=linear_kernel)
            ordinal_weighted_krippendorffs_alpha = krippendorffs_alpha(questions_answers_table,weights_kernel=ordinal_kernel)
            results.append({
                "question_id": qid,
                "score_type": score_col,
                "krippendorffs_alpha_nominal": unweighted_krippendorffs_alpha,
                "krippendorffs_alpha_linear": linear_weighted_krippendorffs_alpha,
                "krippendorffs_alpha_ordinal": ordinal_weighted_krippendorffs_alpha
                })                           
                  
results_df = pd.DataFrame(results)

# Calculate the average Krippendorff's Alpha for each score type
avg_row = results_df.groupby('score_type').mean(numeric_only=True).reset_index()
avg_row['question_id'] = 'average'


results_combined = pd.concat([results_df, avg_row], ignore_index=True)

# Reorder columns to place 'question_id' first
cols = ['question_id'] + [col for col in results_combined.columns if col != 'question_id']
results_combined = results_combined[cols]

# Output the results to a CSV file
output_path = "<KRIPPENDORFFS_ALPHA_FILE>"
results_combined.to_csv(output_path, index=False)

