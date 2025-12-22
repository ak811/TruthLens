# TruthLens

Training-free deepfake image verification via VQA-style probing with LVLMs + LLM reasoning.

**Paper:** Accepted to ICML 2025 — [paper](https://icml.cc/virtual/2025/51033)

---

## Overview

TruthLens reframes fake image detection as a **Visual Question Answering (VQA)** problem. Instead of an opaque binary classifier, it:

1. **Probes** an input image with a set of artifact-focused prompts using a **Large Vision-Language Model (LVLM)** (e.g., Chat-UniVi).
2. **Aggregates** the LVLM’s natural-language answers into a structured evidence summary.
3. **Reasons** over that evidence with an **LLM** to output a final **verdict** (`REAL`/`FAKE`) and a concise **justification**.

The goal is **instance-level, explainable data verification** without detector fine-tuning.

---

## Repository layout

```
truthlens/
  concatinate_jsons.py
  evaluation_gpt.py
  inference_image_chatunivi.py
  requirements.txt
  CNNDetection-master/         # baseline
  DIRE-main/                   # baseline
```

### Key scripts

- `inference_image_chatunivi.py`  
  Runs Chat-UniVi on a dataset folder and writes per-image text answers to JSON.

- `concatinate_jsons.py`  
  Concatenates multiple per-prompt JSON outputs into `combined_descriptions.json`.

- `evaluation_gpt.py`  
  Uses the OpenAI API to turn the combined descriptions into:
  - `{"verdict": "FAKE"|"REAL", "justification": "..."}` per image
  - `analysis_metrics.json` (basic metrics; see note below)

---

## Installation

### 1) Create an environment and install repo dependencies

```bash
conda create -n truthlens python=3.10 -y
conda activate truthlens
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- torch==2.5.1
- tqdm==4.67.1
- Pillow==10.0.1
- openai==0.28.0 (legacy client used by `evaluation_gpt.py`)

### 2) Install Chat-UniVi (required for LVLM probing)

`inference_image_chatunivi.py` imports `ChatUniVi.*`, so you must install Chat-UniVi in the same environment.

One common approach:

```bash
git clone https://github.com/PKU-YuanGroup/Chat-UniVi
cd Chat-UniVi
pip install -e .
cd ..
```

Validate:

```bash
python -c "import ChatUniVi; print('ChatUniVi import OK')"
```

> Note: Chat-UniVi may have extra CUDA / system requirements depending on your setup.

---

## Data format expected by the probing script

`inference_image_chatunivi.py` expects a dataset directory with two subfolders:

```
<dataset_path>/
  fake1000/
    *.png|*.jpg|*.jpeg|*.bmp|*.tiff
  first1000/
    *.png|*.jpg|*.jpeg|*.bmp|*.tiff
```

It will write JSON output files into each subfolder and a combined JSON at the dataset root.

---

## Running TruthLens (probe → aggregate → verdict)

### Step 1: LVLM probing (Chat-UniVi)

Open `inference_image_chatunivi.py` and set:

- `dataset_path = "your path"`
- GPU selection: the script currently hardcodes  
  `os.environ["CUDA_VISIBLE_DEVICES"] = "1"`  
  Change `"1"` to the GPU you want (or remove the line).

Run:

```bash
python inference_image_chatunivi.py
```

This script currently runs **one** prompt:
> “Taking into account the lighting, texture, symmetry, and other features...”

To run the full TruthLens probe set, repeat the probing pass with different `query` strings (see **Prompt set** below) and save each pass to a separate JSON.

### Step 2: Aggregate multiple probe outputs

Edit `concatinate_jsons.py` to point `json_paths` at the JSON files produced by your different prompt runs, then:

```bash
python concatinate_jsons.py
```

Output:
- `combined_descriptions.json`

### Step 3: LLM verdict + explanation (OpenAI API)

Run:

```bash
python evaluation_gpt.py   --description_file combined_descriptions.json   --output_dir outputs/   --api_key YOUR_OPENAI_KEY
```

Outputs:
- `outputs/<image_name>_analysis.json` for each image
- `outputs/analysis_metrics.json`

**Metrics note:** `evaluation_gpt.py` currently computes accuracy as `fake_count / total_images`, which only makes sense if your description file contains *only fake images*. If you mix real and fake images together, you should update the metrics logic to use labels.

---

## Prompt set (TruthLens probes)

Recommended categories (adapt these into separate probe runs):

- Lighting and Shadows  
- Texture and Skin Details  
- Symmetry and Proportions  
- Reflections and Highlights  
- Facial Features and Expression  
- Facial Hair (if applicable)  
- Eyes and Pupils  
- Background and Depth Perception  
- Overall Realism

Tip: save **one JSON per prompt category** so aggregation is clean and debuggable.

---

## Baselines included

### CNNDetection (`CNNDetection-master/`)

CNN-based binary classifier baseline. Includes its own scripts, dataset download helpers, and weights.
Follow `CNNDetection-master/README.md`.

### DIRE (`DIRE-main/`)

Diffusion Reconstruction Error baseline. Includes guided diffusion utilities and its own scripts.
Follow `DIRE-main/README.md`.

---

## Citation

If you use TruthLens, cite the paper:

- **ICML 2025 page:** [paper](https://icml.cc/virtual/2025/51033)

(You can add a BibTeX entry here once you decide the canonical citation format you want in the repo.)

---

## License

This repository includes third-party code under `CNNDetection-master/` and `DIRE-main/`. See those subfolders for their respective licenses and terms.
