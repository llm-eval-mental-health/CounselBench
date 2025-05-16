#!/bin/bash
"""
This script is used to analyze overall reasons for responses with low overall scores (<=2).
"""


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

prompts=(
  "ask_for_toxic_sentence"
  "ask_for_inconsistent_sentence"
  "ask_for_medical_advice_sentence"
)

GEMINI_RESPONSE_FILE=<gemini_response_file_path>
GPT_RESPONSE_FILE=<gpt_response_file_path>
LLAMA_RESPONSE_FILE=<llama_response_file_path>
HUMAN_RESPONSE_FILE=<human_response_file_path>
HUMAN_ANNOTATION_FILE=<human_annotation_file_path>

for model in "${models[@]}"; do
  echo "Running judge for: $model"
  for prompt in "${prompts[@]}"; do
    echo "Running prompt strategy: $prompt"
    python llm_as_judges/automated_judge.py \
        --judge_model_name $model \
        --prompt_strategy $prompt \
        --gemini_response_file $GEMINI_RESPONSE_FILE \
        --gpt_response_file $GPT_RESPONSE_FILE \
        --human_response_file $HUMAN_RESPONSE_FILE \
        --llama_response_file $LLAMA_RESPONSE_FILE \
        --output_dir results/$prompt/$model \
        --human_annotation_file $HUMAN_ANNOTATION_FILE
  done
done