#!/bin/bash
"""
This script is used to analyze overall reasons for responses with low overall scores (<=2).
"""

models=(
    "gpt-4.1"
)

prompt_strategy="classify_overall"

GEMINI_RESPONSE_FILE=<gemini_response_file_path>
GPT_RESPONSE_FILE=<gpt_response_file_path>
LLAMA_RESPONSE_FILE=<llama_response_file_path>
HUMAN_RESPONSE_FILE=<human_response_file_path>
HUMAN_ANNOTATION_FILE=<human_annotation_file_path>

for model in "${models[@]}"; do
    echo "Running judge for: $model"
    python llm_as_judges/automated_judge.py \
        --judge_model_name $model \
        --prompt_strategy $prompt_strategy \
        --gemini_response_file $GEMINI_RESPONSE_FILE \
        --gpt_response_file $GPT_RESPONSE_FILE\
        --human_response_file $HUMAN_RESPONSE_FILE \
        --llama_response_file $LLAMA_RESPONSE_FILE \
        --output_dir results/$prompt_strategy/$model \
        --human_annotation_file $HUMAN_ANNOTATION_FILE
done