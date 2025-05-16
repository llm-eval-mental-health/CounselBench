#!/bin/bash

"""
This script is used to analyze the missed medical advice / incorrect / toxic sentences by LLM judges.
We especailly focus on sentences that are flagged by at least two human annotators.
"""

metrics=(
    "medical_advice"
    "factual_consistency"
    "toxicity"
)

for metric in "${metrics[@]}"; do
    python -m pdb -c llm_as_judges/case_study.py \
        --model_name "gpt-3.5-turbo" "gpt-4-0613" "meta-llama/Llama-3.1-70B-Instruct" "meta-llama/Llama-3.3-70B-Instruct" "claude-3-5-sonnet-20241022" "claude-3-7-sonnet-20250219" "gemini-1.5-pro" "gemini-2.0-flash" \
        --metric $metric 
done