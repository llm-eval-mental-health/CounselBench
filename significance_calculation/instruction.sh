#Analysis Process Overview
Step 1: Run get_data.py
This script will read data from the original JSON file, extract the scores from different annotators for each question, and retain the source of the response (e.g., human or a specific LLM) for each question.
Step 2: Run get_human_evaluation.py
This script calculates the mean score of the five annotators for each metric (e.g., "Overall", "Empathy", etc.) for the same question and response source, preparing the data for subsequent significance calculations.
Step 3: Run human_evaluation.py
This script ensures the data format is consistent with the model evaluation files, making it easier to compare with the model evaluation results.
Step 4: Run wilcoxon_calculate.py
This script performs the Wilcoxon Signed-Rank test to calculate the statistical significance between human evaluations and model evaluations