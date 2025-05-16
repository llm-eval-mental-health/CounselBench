import pandas as pd
from scipy.stats import wilcoxon

# List of LLMs and models for comparison
llms = ['claude-3-5-sonnet-20241022', 'claude-3-7-sonnet-20250219',
        'gemini-1.5-pro', 'gemini-2.0-flash', 'gpt-4-0613',"gpt-3.5-turbo",
        'meta-llama-Llama-3.3-70B-Instruct', 'meta-llama-Llama-3.1-70B-Instruct']

models = ['gemini_1.5_pro', 'gpt4', "human", "llama3.3"]

wilcoxon_results = []

for model in models:
    df_human = pd.read_csv(
        f'<{MODEL}_EVAlUATED_BY_HUMAN_FILE_WITH_THE_SAME_FORMAT_AS_THE_LLM_EVALUATION>',
        encoding='utf-8'
    )
    
    for llm in llms:
        try:
           
            df_model = pd.read_csv(
                f'<{MODEL}_EVALUATED_BY_{LLM}_FILE>'
            )
        except FileNotFoundError:
            print(f"File does not exist: {llm}")
            continue


        common_ids = set(df_human['questionID']) & set(df_model['questionID'])
        df_human_filtered = (
            df_human[df_human['questionID'].isin(common_ids)]
            .sort_values(by="questionID")
            .reset_index(drop=True)
        )
        df_model_filtered = (
            df_model[df_model['questionID'].isin(common_ids)]
            .sort_values(by="questionID")
            .reset_index(drop=True)
        )


        columns = ["Overall", 'Empathy', 'Specificity', 'Toxicity']

        for col in columns:
            x = df_human_filtered[col].astype(float)  # human scores
            y = df_model_filtered[col].astype(float)  # LLM scores
            try:
                statistic, p_value = wilcoxon(x, y)
                significant = p_value < 0.05  
            except ValueError as e:

                print(f"The Wilcoxon test cannot be calculated {model} vs {llm}, {col}: {e}")
                statistic, p_value, significant = None, None, None

            wilcoxon_results.append({
                "model": model,
                "llm": llm,
                "col": col,
                "wilcoxon_statistic": round(statistic, 3) if statistic is not None else None,
                "wilcoxon_p": p_value if p_value is not None else None,
                "significant": significant
            })


wilcoxon_df = pd.DataFrame(wilcoxon_results)
wilcoxon_df.to_csv(
    "<WILCOXON_FILE>",
    index=False
)

