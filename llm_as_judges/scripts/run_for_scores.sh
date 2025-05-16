#!/bin/bash

models=(
  "gpt-3.5-turbo"
  "gpt-4-0613"
  "claude-3-5-sonnet-20241022"
  "claude-3-7-sonnet-20250219"
  "gemini-1.5-pro"
  "gemini-2.0-flash"
  "meta-llama/Llama-3.3-70B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
)

GEMINI_RESPONSE_FILE=<gemini_response_file_path>
GPT_RESPONSE_FILE=<gpt_response_file_path>
LLAMA_RESPONSE_FILE=<llama_response_file_path>
HUMAN_RESPONSE_FILE=<human_response_file_path>
HUMAN_ANNOTATION_FILE=<human_annotation_file_path>

for model in "${models[@]}"; do
    echo "Running judge for: $model"
    python llm_as_judges/automated_judge.py \
        --judge_model_name $model \
        --prompt_strategy single_eval \
        --gemini_response_file $GEMINI_RESPONSE_FILE \
        --gpt_response_file $GPT_RESPONSE_FILE \
        --human_response_file $HUMAN_RESPONSE_FILE \
        --llama_response_file $LLAMA_RESPONSE_FILE \
        --output_dir results/single_eval/$model \
        --human_annotation_file  $HUMAN_ANNOTATION_FILE
done