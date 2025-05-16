#!/bin/bash

responder_models=(
    "gpt-3.5-turbo"
    "gpt"
    "claude-3-5-sonnet-20241022"
    "claude-3-7-sonnet-20250219"
    "gemini-2.0-flash"
    "gemini"
    "meta-llama/Llama-3.3-70B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
)
judge_models=(
    "gpt-4.1"
)

input_file="<adverarial_input_file_path>"
example_file="<adverarial_example_file_path>" 
base_output_dir="<output_dir_path>"

for responder in "${responder_models[@]}"; do
  for judge in "${judge_models[@]}"; do
    echo "► Running responder=$responder | judge=$judge"
    python run_adversarial/run_adversarial_questions.py \
      --input_file       "$input_file" \
      --example_file     "$example_file" \
      --responder_model_name "$responder" \
      --judge_model_name "$judge" \
      --output_dir       "$base_output_dir/${responder}_by_${judge}"
  done
done