# Data preparation

No images are distributed with this repository. The paper (§3.1) evaluates on:

| Subset | Label | Size | Source | Notes |
|--------|-------|------|--------|-------|
| `real` | REAL | 1,000 | FFHQ (Karras et al., 2019) | The released script called this folder `first1000`, suggesting the first 1,000 FFHQ images; the exact file list and resolution are not recorded. |
| `ldm` | FAKE | 1,000 | Latent Diffusion Model faces (Rombach et al., 2022) | The generating checkpoint and sampling settings are not recorded. The released script called this folder `fake1000`. |
| `progan` | FAKE | 1,000 | ProGAN images from ForgeryNet (He et al., 2021) | The ForgeryNet split/selection is not recorded. |

FFHQ and ForgeryNet have their own licences and access procedures; obtain them from the
original providers. See [AUDIT.md §B6](AUDIT.md) for the provenance gaps. If you are an
author, please record the exact file lists (for example, as manifests) so that others can
recreate the evaluation set.

## 1. Folder layout

Any layout works as long as each subset is a folder of images. For example:

```
data/
  ffhq_first1000/     *.png
  ldm_fake1000/       *.png
  progan_fake1000/    *.png
```

Accepted extensions: `.png .jpg .jpeg .bmp .tif .tiff .webp` (case-insensitive). Add
`--recursive` to include sub-folders.

## 2. Build the manifest

```bash
truthlens manifest \
  --real data/ffhq_first1000 \
  --fake ldm=data/ldm_fake1000 \
  --fake progan=data/progan_fake1000 \
  --out data/manifest.jsonl
```

Each line of `data/manifest.jsonl` has the form:

```json
{"id": "ldm/00012.png", "path": "ldm_fake1000/00012.png", "label": "FAKE", "subset": "ldm"}
```

* `id` = `<subset>/<path relative to the subset folder>`; ids are unique even when real
  and fake files share names.
* `path` is stored relative to the manifest file and resolved against it on load.
* Files are listed in sorted order. `--limit-per-subset N` keeps the first N per subset.

Metrics are computed per fake subset against **all** REAL images, plus an `all` row.

## 3. Folders for the baselines

CNNDetection and DIRE expect `<dir>/0_real` and `<dir>/1_fake`. Build them from the same
manifest so that all methods see identical images:

```bash
python scripts/make_baseline_folders.py --manifest data/manifest.jsonl --fake-subset ldm    --out data/baselines/ldm
python scripts/make_baseline_folders.py --manifest data/manifest.jsonl --fake-subset progan --out data/baselines/progan
```

Files are symlinked (use `--copy` to copy). For DIRE, first convert both folders into DIRE
maps with `baselines/DIRE/guided-diffusion/compute_dire.py` (see `baselines/README.md`).

## 4. Importing outputs from the released scripts

If you still have per-category JSON files from the released `inference_image_chatunivi.py`
(for example `eyes_chat_univi_results_fake1000.json`, keyed by absolute image path), you
can import them instead of re-running the LVLM:

```bash
truthlens import-legacy eyes_chat_univi_results_fake1000.json  \
  --category eyes --subset ldm --config configs/truthlens_chatunivi.yaml
```

Keys are matched to manifest images by **file name** within `--subset`. Suggested mapping
from released file stems to category keys (`faceattributes` is inferred from the name;
please confirm):

| Released stem | Category key |
|---------------|--------------|
| `eyes` | `eyes` |
| `faceattributes` | `facial_features` |
| `facialhair` | `facial_hair` |
| `realism_2` | `overall_realism` |
| `reflections` | `reflections` |
| `symmetry_2` | `symmetry` |
| `texture` | `texture` |

Then run `truthlens aggregate`, `judge` and `evaluate` as usual. Use
`--set aggregate.categories=released_7` because lighting and background were not produced.
