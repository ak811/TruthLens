#!/usr/bin/env bash
# End-to-end CPU smoke test (mock LVLM + mock judge). Run from the repository root.
set -euo pipefail
rm -rf data/smoke outputs/smoke_test
python scripts/make_smoke_data.py --root data/smoke --n 6
truthlens manifest --real data/smoke/real --fake ldm=data/smoke/ldm --fake progan=data/smoke/progan \
    --out data/smoke/manifest.jsonl
truthlens run --config configs/smoke_test.yaml
truthlens yesno --config configs/smoke_test.yaml
python - <<'PY'
import json
m = json.load(open("outputs/smoke_test/metrics.json"))["results"]
y = json.load(open("outputs/smoke_test/yesno_metrics.json"))["results"]
for name, res in (("pipeline", m), ("yes/no", y)):
    for subset in ("ldm", "progan", "all"):
        assert res[subset]["accuracy"] == 1.0 and res[subset]["n_invalid"] == 0, (name, subset, res[subset])
print("SMOKE TEST PASSED")
PY
