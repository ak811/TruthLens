# Baselines: CNNDetection and DIRE

This directory vendors the two detectors compared against TruthLens in Tables 1 and 3.
Both are third-party code; see their own READMEs for their methods.

| Directory | Upstream | Paper | Licence |
|-----------|----------|-------|---------|
| `CNNDetection/` | https://github.com/PeterWang512/CNNDetection (`master`) | Wang et al., *CNN-generated images are surprisingly easy to spot... for now*, CVPR 2020 | CC BY-NC-SA 4.0 (`CNNDetection/LICENSE.txt`) |
| `DIRE/` | https://github.com/ZhendongWang6/DIRE (`main`) | Wang et al., *DIRE for Diffusion-Generated Image Detection*, ICCV 2023 | `guided-diffusion/` is MIT (`DIRE/guided-diffusion/LICENSE`). The DIRE repository ships **no licence file**; check redistribution terms with its authors before publishing. |

(These directories were previously named `CNNDetection-master/` and `DIRE-main/` at the
repository root.)

## Provenance and modifications

The vendored files were restored from the upstream repositories listed above and match the
previously released copies, except for the changes below. The TruthLens-specific
evaluation scripts are `CNNDetection/inference.py` and `DIRE/inference.py` (not upstream).

TruthLens additions and edits:

* `CNNDetection/inference.py`, `DIRE/inference.py`: TruthLens evaluation scripts, with
  bug fixes (see `docs/AUDIT.md`, C15–C16), JSON output (`--output_json`), per-class
  accuracies, and input validation.
* `DIRE/test.sh`: `CKPT="lsun_adm.pth"`, as in the previously released copy.

Upstream bug fixes (see `docs/AUDIT.md`, C14 and C17–C21):

* `CNNDetection/data/` restored. `train.py`, `eval.py` and `validate.py` import it; it was
  missing from the released copy.
* `np.Inf` → `np.inf` (`CNNDetection/earlystop.py`, `DIRE/utils/earlystop.py`).
* `np.int` → `np.int64` (`DIRE/guided-diffusion/guided_diffusion/resample.py`).
* Missing comma in `str2bool` (`DIRE/utils/utils.py`).
* Swapped arguments in the `FileNameDataset` call (`DIRE/utils/datasets.py`).
* Explicit errors for `--num_samples <= 0` and an unset `CUDA_VISIBLE_DEVICES`
  (`DIRE/guided-diffusion/compute_dire.py`).
* Removed an unused `IPython` import (`CNNDetection/networks/lpf.py`).

The upstream READMEs mention `demo.py`/`demo_dir.py`, which are not vendored here; use
`inference.py` instead.

## Preparing inputs

Build `0_real/1_fake` folders from the TruthLens manifest so that the baselines see exactly
the same images:

```bash
python scripts/make_baseline_folders.py --manifest data/manifest.jsonl --fake-subset ldm    --out data/baselines/ldm
python scripts/make_baseline_folders.py --manifest data/manifest.jsonl --fake-subset progan --out data/baselines/progan
mkdir -p outputs/baselines
```

## CNNDetection

Weights: the `.pth` files in `CNNDetection/weights/` are Git LFS pointers. Run
`git lfs pull` in a clone of this repository, or `bash weights/download_weights.sh` from
inside `CNNDetection/`. `inference.py` detects un-fetched pointers and says so.

```bash
cd baselines/CNNDetection
python inference.py -d ../../data/baselines/ldm    -m weights/blur_jpg_prob0.5.pth --output_json ../../outputs/baselines/cnndetection_ldm.json
python inference.py -d ../../data/baselines/progan -m weights/blur_jpg_prob0.5.pth --output_json ../../outputs/baselines/cnndetection_progan.json
```

By default, images are resized to 224×224 (the setting of the released script). Add
`--no_resize` for the upstream protocol, which forces `--batch_size 1`. The paper does not
state which weight file (`blur_jpg_prob0.5` or `blur_jpg_prob0.1`) was used.

## DIRE

DIRE classifies *DIRE maps*, i.e. |x − reconstruction(x)| computed with a pretrained
diffusion model, not raw images.

1. Download a guided-diffusion model (for example `256x256_diffusion_uncond.pt`; links in
   `DIRE/guided-diffusion/README.md`) and a DIRE classifier checkpoint (links in
   `DIRE/README.md`). The paper does not record which DIRE checkpoint was used; `test.sh`
   references `lsun_adm.pth`, which was trained on LSUN bedrooms, not faces.
2. Compute DIRE maps for each class folder. Set `--num_samples` to the number of images
   in the folder:

   ```bash
   cd baselines/DIRE/guided-diffusion
   export CUDA_VISIBLE_DEVICES=0
   MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
   SAMPLE_FLAGS="--batch_size 16 --num_samples 1000 --timestep_respacing ddim20 --use_ddim True"
   for C in 0_real 1_fake; do
     mpiexec -n 1 python compute_dire.py --model_path models/256x256_diffusion_uncond.pt $MODEL_FLAGS $SAMPLE_FLAGS \
       --images_dir ../../../data/baselines/ldm/$C \
       --recons_dir ../../../data/dire/recons/ldm/$C \
       --dire_dir   ../../../data/dire/dire/ldm/$C
   done
   ```

3. Classify:

   ```bash
   cd baselines/DIRE
   python inference.py -d ../../data/dire/dire/ldm --input_type dire -m /path/to/dire_checkpoint.pth \
       --output_json ../../outputs/baselines/dire_ldm.json
   ```

`--input_type raw` runs the classifier directly on images. This is **not** the DIRE
protocol, and the script prints a warning. It exists only to reproduce a raw-image
evaluation if one was used (see `docs/AUDIT.md`, B5).

## Metrics

Both scripts report `accuracy`, `real_accuracy`, `fake_accuracy`, `precision`, `recall`,
`f1` (FAKE = positive), `roc_auc` and `ap` from the sigmoid probabilities, along with the
confusion counts. The ROC-AUC here is threshold-free; TruthLens's Table 1 value is
computed from hard verdicts (`docs/AUDIT.md`, B2).
