# Reproducibility guide

This guide maps each result of the paper to a command, lists the environments and assets
required, and documents what is not recorded and may therefore cause differences. **None
of the paper's numbers have been re-run as part of this refactor** (see
[AUDIT.md §E](AUDIT.md)).

## 1. Environments

The pipeline has a GPU-bound stage (LVLM probing) and CPU-only stages (aggregation, LLM
judge, evaluation). The core package has no PyTorch dependency, so it can be installed
next to any LVLM stack without changing that stack's pinned versions.

| Env | Purpose | Setup |
|-----|---------|-------|
| **A. core** | aggregate, judge, evaluate, tests | `pip install -e ".[judge]"` (tests: `pip install -e ".[dev]"`), Python ≥ 3.9 |
| **B. Chat-UniVi** | probing with the paper's main LVLM | Install Chat-UniVi following its README (`git clone https://github.com/PKU-YuanGroup/Chat-UniVi && pip install -e Chat-UniVi`, which pins its own torch/transformers), then `pip install -e /path/to/TruthLens` in the same environment |
| **C. HF LVLMs** | BLIP-2, LLaVA-1.5 (Table 2) | `pip install -e ".[hf,judge]"`. CogVLM's remote code requires an older `transformers`; follow the model card |
| **D. baselines** | CNNDetection, DIRE | Separate environments per `baselines/*/requirements.txt`; DIRE upstream uses Python 3.9 and `torch==2.0.0+cu117`, plus `mpi4py` for `compute_dire.py` |

The previous README pinned `torch==2.5.1` and `openai==0.28.0` for the TruthLens scripts.
`openai==0.28` is incompatible with this code (it requires `openai>=1.0`). Chat-UniVi pins
its own torch version, and forcing a different one into its environment is not recommended.

Hardware: probing needs a CUDA GPU able to hold the chosen LVLM (Chat-UniVi is a
7B-parameter model loaded in half precision). Everything else runs on a CPU.

Credentials: the judge reads `OPENAI_API_KEY` from the environment. `OPENAI_BASE_URL` may
point to an OpenAI-compatible server.

## 2. Workload

With the paper's data (3,000 images), a full run issues 3,000 × 9 = 27,000 LVLM
generations (up to 1,024 new tokens each) and 3,000 judge calls. The Yes/No baseline issues
3,000 LVLM generations per model. All long-running stages append to JSONL files after every
item and **resume** when re-run; failed items are retried on re-run.

## 3. Experiment map

Prerequisite: `data/manifest.jsonl` ([DATA.md](DATA.md)).

| Paper result | Command | Output |
|--------------|---------|--------|
| Table 2, "Prompts + LLM", ChatUniVi; TruthLens rows of Tables 1 and 3 | `truthlens run --config configs/truthlens_chatunivi.yaml` | `outputs/chatunivi_gpt4/metrics.json` |
| Table 2, "Yes or No Question", ChatUniVi | `truthlens yesno --config configs/truthlens_chatunivi.yaml` | `.../yesno_metrics.json` |
| Table 2, BLIP-2 / LLaVA-1.5 / CogVLM rows | `truthlens run` and `truthlens yesno` with `configs/table2_{blip2,llava15,cogvlm}.yaml` | `outputs/<model>_gpt4/` |
| Table 4 (per-category ablation) | `bash scripts/run_ablation_table4.sh configs/truthlens_chatunivi.yaml outputs/chatunivi_gpt4` | `outputs/chatunivi_gpt4_ablation/<category>/metrics.json` |
| Released pipeline, for comparison (7 categories, `" \| "` concatenation, `gpt-3.5-turbo`) | `truthlens run --config configs/released_pipeline.yaml` | `outputs/released_pipeline/` |
| CNNDetection rows of Tables 1 and 3 | see `baselines/README.md` | JSON via `--output_json` |
| DIRE rows of Tables 1 and 3 | see `baselines/README.md` | JSON via `--output_json` |

`scripts/reproduce_paper.sh` chains the TruthLens commands.

Mapping of metric keys to the paper's columns:

| Paper | `metrics.json` key (in `results.<ldm|progan>`) |
|-------|------------------------------------------------|
| Table 2 "Real (%)" | `real_accuracy` (identical in both rows, since the real set is shared) |
| Table 2 "Fake (%)" LDM / ProGAN | `fake_accuracy` |
| Table 1 AUC (TruthLens) | `auc_hard` (= `balanced_accuracy`, see AUDIT B2) |
| Table 3 Precision / Recall / F1 | `precision` / `recall` / `f1` (FAKE = positive) |
| Table 4 accuracy | `accuracy` of the `all` row is one plausible reading; the paper does not specify the image set |

The stages can also be run separately:

```bash
truthlens probe     --config configs/truthlens_chatunivi.yaml   # GPU environment
truthlens aggregate --config configs/truthlens_chatunivi.yaml   # any environment
truthlens judge     --config configs/truthlens_chatunivi.yaml
truthlens evaluate  --config configs/truthlens_chatunivi.yaml
```

Any configuration value can be overridden, e.g. `--set judge.model=gpt-4-0613 --set
judge.temperature=0`. The fully resolved configuration is written to
`<output_dir>/config.resolved.yaml`. Package versions, the git commit and the LVLM/judge
settings of each stage are written to `<output_dir>/run_info.json`.

## 4. Randomness and determinism

* `seed` (default 0) sets a per-item seed `stable_seed(seed, image_id, category)` before
  every LVLM generation. Results therefore do not depend on processing order or on
  interruptions. The Yes/No baseline uses `stable_seed(seed, image_id, "yesno")`.
* GPU kernels can still be non-deterministic. For fully greedy decoding, use
  `--set probe.generation.do_sample=false`. This differs from the released script,
  which sampled at T = 0.2.
* The OpenAI API is not deterministic. `judge.temperature` defaults to the API default,
  as in the released code; `judge.seed` is forwarded to the API (best effort). The returned
  model identifier and `system_fingerprint` are stored with every verdict. Hosted model
  aliases such as `gpt-4` may change or be retired over time.
* No step involves training; the method is training-free.

## 5. Intermediate files

| File | One record per | Key fields |
|------|----------------|------------|
| `probes.jsonl` | image × category | `id, subset, label, category, prompt, response, error, seed, backend, model_path` |
| `summaries.jsonl` | image | `id, mode, categories, missing_categories, summary` |
| `verdicts.jsonl` | image | `verdict, confidence, justification, parse_status, raw_response, error, judge_model, returned_model, system_fingerprint` |
| `metrics.json` | run | `results.{ldm,progan,all}.{real_accuracy, fake_accuracy, balanced_accuracy, auc_hard, precision, recall, f1, n, n_invalid, ...}` |

A verdict is *invalid* if the API call failed, the reply could not be parsed, or the image
had no usable probe answers. Invalid predictions count as errors by default
(`evaluate.invalid_policy: incorrect`) and are always reported as `n_invalid`.

## 6. Known differences from the paper

Settings the paper does not record, and choices where the paper and the released code
disagree, are listed in [AUDIT.md](AUDIT.md). The ones most likely to change the numbers
are:

1. the set of aggregated categories (9 in the paper, 7 in the released code; A2);
2. the aggregation format (A3);
3. the judge model and snapshot (A4);
4. decoding parameters (A6);
5. LVLM checkpoints for the Table 2 baselines and the Yes/No prompt wording (A7);
6. the exact image lists (B6);
7. the baseline protocols (B5).

Use `configs/released_pipeline.yaml` to test whether the released code, rather than the
paper's description, reproduces the published numbers.

## 7. Verifying an installation

```bash
pip install -e ".[dev]"
pytest -q                     # unit + integration tests (torch-dependent baseline tests are skipped without torch)
bash scripts/smoke_test.sh    # CPU end-to-end run with mock LVLM and mock judge; prints SMOKE TEST PASSED
```

The smoke test checks plumbing only (ids, labels, aggregation, parsing, metrics). Its 100 %
accuracy is by construction and has no scientific meaning.
