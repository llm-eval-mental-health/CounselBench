#Analysis Process Overview
Step 1: Run get_data.py
This script will read data from the original JSON file, extract the scores from different annotators for each question, and retain the source of the response (e.g., human or a specific LLM) for each question.
Step 2: Run KA_calculate.py
This script will use the data obtained from get_data.py to calculate the agreement of five metrics (e.g., "Overall", "Empathy", etc.) using Krippendorff's Alpha.