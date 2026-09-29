#!/usr/bin/env bash
# Table 4 (per-category ablation): judge each probe category in isolation, reusing
# the probe answers of a completed main run. Usage:
#   bash scripts/run_ablation_table4.sh configs/truthlens_chatunivi.yaml outputs/chatunivi_gpt4
# The paper does not state on which image set the Table 4 accuracy was computed;
# the 'all' row of each metrics.json (all real + all fake images) is one option.
set -euo pipefail
CONFIG=${1:?config}; MAIN_OUT=${2:?output dir of the main run (contains probes.jsonl)}
for CAT in lighting texture symmetry reflections facial_features facial_hair eyes background overall_realism; do
  OUT="${MAIN_OUT}_ablation/${CAT}"
  truthlens aggregate --config "$CONFIG" --output-dir "$OUT" \
      --set aggregate.categories="[$CAT]" --set aggregate.probes_file="$MAIN_OUT/probes.jsonl"
  truthlens judge     --config "$CONFIG" --output-dir "$OUT"
  truthlens evaluate  --config "$CONFIG" --output-dir "$OUT"
done
