#!/usr/bin/env bash
# Commands to regenerate the TruthLens numbers of Tables 1-3 (ChatUniVi + GPT-4 judge)
# and the Table 2 baselines. Requires a GPU environment with Chat-UniVi, an OpenAI key,
# and data/manifest.jsonl (docs/DATA.md). See docs/REPRODUCIBILITY.md before running:
# several settings used for the paper are not recorded and results may differ.
set -euo pipefail
: "${OPENAI_API_KEY:?export OPENAI_API_KEY first}"

# Table 2, last row (Prompts + LLM, ChatUniVi) and TruthLens rows of Tables 1 and 3
truthlens run   --config configs/truthlens_chatunivi.yaml
# Table 2, 'Yes or No Question' row for ChatUniVi
truthlens yesno --config configs/truthlens_chatunivi.yaml

# Other Table 2 rows (checkpoints are assumptions - edit the configs)
for M in blip2 llava15 cogvlm; do
  truthlens run   --config configs/table2_$M.yaml
  truthlens yesno --config configs/table2_$M.yaml
done

# Table 4 ablation
bash scripts/run_ablation_table4.sh configs/truthlens_chatunivi.yaml outputs/chatunivi_gpt4
