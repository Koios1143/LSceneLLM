# XR-QA with the released LSceneLLM checkpoint

Evaluates the released checkpoint (HF `Hoyard/LSceneLLM`, `release.pth`) on XR-QA val (HF
dataset `Hoyard/XR-Scene`, 2994 questions, 24 HM3D scans, 268 regions). The evaluation is
the repository's own (`scripts/eval.sh`: `main_ALLM.py --config cfgs/XR-QA-Evaluation.yaml
--test`). Scoring is its own `validate` (`utils/capeval` BLEU / CIDEr / ROUGE-L / METEOR on the
raw decoded text, keyed by episode id). The repository only evaluates XR-QA: it has no
XR-SceneCaption dataset class or prompt.

```
bash extra_eval/build_env.sh   # once, on a GPU node
bash extra_eval/run_xrqa.sh    # OpenScene features of the val regions, then inference + scoring
```

`env.sh.example` lists the variables. `run_xrqa.sh` lists the expected data / checkpoint
layout: `data/` and `ckpts/` at the repository root, the paths the upstream code uses.

## Changes needed to run it on current hardware (none in the upstream files)

* **Environment** (`build_env.sh`). The README's python 3.7 / old torch has no Blackwell kernels.
  This uses python 3.10, torch 2.7.1+cu128 and transformers 4.40.2.
  `models/modeling_llama.py` is a copy of the transformers 4.36/4.37 Llama.
  pointnet2_ops is built with its hard-coded arch list removed.
  KNN_CUDA (github.com/unlimblue/KNN_CUDA) is no longer online. `knn_cuda_torch/` is an exact torch
  drop-in with the same interface; only `models/dvae.py` (DGCNN) uses it, and that code is not on
  the OpenScene encoder path.
* **OpenScene features** (`openscene_features.py`). This is the computation of
  `models/openscene/run/preprocess.py`: distill features, MinkUNet18A, `scannet_openseg.pth.tar`,
  2 cm voxels, the same saved dict. It processes only the regions the annotation files use and
  skips the CLIP / SharedArray imports, which the distill path never uses.
* **Launcher** (`torchrun_main.py`):
  * `--local_rank` is passed from torchrun's `LOCAL_RANK`.
  * The LLM is loaded with eager attention. The adaptive dense-token selection exists only in
    `LlamaAttention`, whereas transformers ≥ 4.36 would pick `LlamaSdpaAttention` by default.
  * There is one item per process. In evaluation the model stores the whole batch's dense
    tokens on the attention layers but generates each item alone, so a larger per-GPU batch
    fails, or attends to another item's tokens. Generation is per item anyway.
* `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`: the data loader `torch.load`s SceneVerse `.pth` files
  that hold numpy arrays.
* METEOR needs `utils/capeval/meteor/data/paraphrase-en.gz`, which the repository lacks. It is the
  METEOR 1.5 file, the same one as in LL3DA's identical `utils/capeval`.

Speed: about 15 s per question on one RTX PRO 6000. The upstream greedy decoding recomputes
the whole sequence for every token, with no KV cache. On 2 GPUs the run takes about 6.5 h.

## Results

XR-QA val, released checkpoint (`results/hd_scene_qa_-1_qa_scores.json`; per-question predictions,
references and scores in `results/hd_scene_qa_-1_qa_pred_gt_val.json`):

| | CIDEr | ROUGE-L | METEOR | BLEU-4 |
|---|---|---|---|---|
| this run | 116.32 | 38.06 | 19.23 | 16.49 |
| paper Table 1, LSceneLLM, as printed (header CIDEr / METEOR / ROUGE) | 117.21 | ROUGE 19.30 | METEOR 38.18 | – |

CIDEr agrees within 0.9. Our ROUGE-L (38.06) matches the paper's "METEOR" column (38.18), and our
METEOR (19.23) matches its "ROUGE" column (19.30). The paper's METEOR and ROUGE columns are
therefore swapped, at least for XR-QA: the upstream `validate` logs ROUGE-L before METEOR.

The scorer keys predictions by `HM3D#scan#region#question`, so the 2994 val questions collapse
to 2981 keys (13 repeated question texts in a region); this is the upstream behaviour.

# ScanNet-family benchmarks with the released LSceneLLM checkpoint

The same checkpoint (`ckpts/release.pth`) on **Scan2Cap**, **ScanQA**, **SQA3D (test)**,
**Real-3DQA** and **Beacon3D QA** (ScanNet, 3RScan, MultiScan). The checkpoint was trained on
LL3DA's data (ScanRefer / Nr3D captions, ScanQA, 3D-LLM) plus XR-Scene. As with LL3DA,
**ScanRefer, Multi3DRefer and Beacon3D grounding are not evaluated**: the model has no
grounding head, and its only way to give a location is to write a box as text, which nothing
in the release evaluates.

```bash
FAST=1 bash extra_eval/run_infer.sh   # GPU: OpenScene features of the scans, then inference (one process per GPU)
bash extra_eval/score.sh              # CPU: own + aligned scoring; Beacon3D QA needs OPENAI_API_KEY and JUDGE_ENV
```

`env.sh.example` lists the variables (`EXTRA_ANNO`, `SCANNET_PCD`, `RSCAN_PCD`, `MULTISCAN_PCD`,
`MASK3D_DIR`, `JUDGE_ENV`).

## Protocol

Only the XR-QA loader is released, so the ScanNet inputs are built the way that loader builds
them (`extra_eval/infer.py` documents every choice):

* **Points and features.** Each scan is a SceneVerse point cloud (ScanNet / MultiScan
  `pcd_with_global_alignment`, 3RScan `pcd-align.pth`). Its OpenScene features come from
  `openscene_features.py --pcds`, the same computation as for XR. All points are moved into
  the unit sphere (`pc_norm`). 160k points are drawn for the dense tokens and 40k for the sparse
  ones. The draw is seeded by the scene name, so every question about a scene sees the same points.
* **Prompts.** QA uses `TASK_PROMPT['scene_qa'][0]` with the question lower-cased, as LL3DA's
  ScanQA loader does. SQA3D and Real-3DQA put `<situation> <question>` into the same prompt. No
  visual prompt is given. Answers are greedy-decoded up to 20 tokens, the hard-coded length.
  Scan2Cap uses `TASK_PROMPT['object_caption'][0]` with a **click prompt** at the centre of the
  Mask3D proposal that has the highest box IoU to the object. The proposals are LEO's released
  `mask.npz`, first 50 per scene, on the same SceneVerse points. Captions go up to 48 tokens.
  There is one item per ScanRefer val object (2068).
* **Decoding.** The model's own `forward(..., eval=True)` under fp16 autocast, as `validate`
  runs it. `FAST=1` (`fast_flex.py`) adds a KV cache to the greedy decoder. The adaptive
  dense-token attention is causal row by row, so cached keys and values give the same result.
  Check against the repository's decoder: in fp32, 80/80 outputs are identical (40 Scan2Cap,
  40 ScanQA). Under fp16 autocast, 59/60 are identical; the one difference is a near-tie on the
  first token. The QA step drops from 1.29 to 0.23 s per question. The whole run takes about 25
  minutes on 2 GPUs. A full run with the repository's decoder (3 h 53 min) gives the same
  output for 99.0% of Scan2Cap objects (2047/2068), 99.8% of ScanQA (4666/4675), 99.9% of
  SQA3D (3514/3519) and Real-3DQA debiased (1484/1485), and 100% of Real-3DQA rotation and
  Beacon3D QA.

## Results (FAST decoding)

**own** = LSceneLLM's `utils/capeval` on the raw text (no PTB), as its `validate` does.
Scan2Cap is scored as LL3DA's `evaluate_densecap`: references are cut to 30 tokens and wrapped
in `sos … eos`; an object scores `sos eos` when its proposal IoU is not above the threshold.
**aligned** = the cross-baseline scorer `aligned/aligned_eval.py`. Scan2Cap: PTB + sos/eos on
both sides, untruncated references. ScanQA: `clean_answer` + PTB.

| Benchmark | Metric | own | aligned | paper |
|---|---|---:|---:|---:|
| ScanQA val (4675) | CIDEr | 74.31 | 74.31 | 88.24 |
| | ROUGE-L | 37.36 | 37.36 | 40.82 |
| | METEOR | 14.97 | 15.08 | 17.95 |
| | BLEU-4 | 10.69 | 10.69 | – |
| | EM / EM-R | 16.86 / 34.97 | 20.49 / 34.97 | – |
| Scan2Cap (2068 objects) | CIDEr@0.5 | 4.06 | 29.46 | – |
| | BLEU-4@0.5 | 19.55 | 26.73 | – |
| | METEOR@0.5 | 18.24 | 22.54 | – |
| | ROUGE-L@0.5 | 44.33 | 50.87 | – |
| | CIDEr@0.25 | 4.44 | 32.12 | – |
| SQA3D test (3519) | EM / EM-R | 37.74 / 41.92 | 37.74 / 41.92 | – |

Paper = LSceneLLM Table 2 (ScanQA val; it is the paper's only ScanNet benchmark). Files:
[`results/scanqa/`](results/scanqa), [`results/scan2cap/`](results/scan2cap),
[`results/sqa3d_test/`](results/sqa3d_test), [`results/aligned/`](results/aligned).

SQA3D per type (EM): what 29.73 · is 51.69 · how 32.26 · can 55.62 · which 28.49 · others 37.46.

Real-3DQA (official scripts, [`results/real3dqa/`](results/real3dqa)):

| Split | EM | EM-R |
|---|---:|---:|
| debiased_test (1485) | 20.27 | 25.25 |
| rotation 0 / 90 / 180 / 270 (750 each) | 9.33 / 8.13 / 8.27 / 8.93 | 14.27 / 13.20 / 13.47 / 12.67 |

| Rotation consistency | One | Two | Three | Four | VRS |
|---|---:|---:|---:|---:|---:|
| | 42.27 | 9.47 | 1.60 | 0.27 | 13.40 |

Beacon3D QA (judge gpt-4o-mini-2024-07-18, [`results/beacon3d_*`](results)):

| Domain | Case | Object | class | appearance | geometry | spatial | existence | functionality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ScanNet (1155) | 45.71 | 9.35 | 18.92 | 40.99 | 53.96 | 48.28 | 55.92 | 33.33 |
| 3RScan (816) | 41.18 | 4.04 | 32.61 | 30.80 | 43.64 | 45.96 | 47.39 | – |
| MultiScan (477) | 28.93 | 3.77 | 18.15 | 27.74 | 45.83 | 24.82 | 37.29 | – |
| pooled (2448) | 40.93 | 6.50 | 21.58 | 35.11 | 49.40 | 42.72 | 48.95 | 33.33 |
| mean of domains | 38.61 | 5.72 | 23.22 | 33.17 | 47.81 | 39.69 | 46.87 | 33.33 |

Notes:

* **ScanQA is 14 CIDEr below the paper.** It does not depend on the prompt choices: keeping the
  question's case gives 74.21 CIDEr, and `scene_qa[1]` gives 74.71
  ([`results/ablations/`](results/ablations)). The XR-QA result of the same pipeline matches
  the paper (116.3 vs 117.2). The released checkpoint is the XR-QA one, which may differ from
  the paper's Table 2 model. In FLEX mode the scene tokens carry no coordinates, so the
  coordinate normalisation cannot explain the gap.
* **Scan2Cap own CIDEr is very low** (4.1) next to aligned (29.5). The model writes
  punctuation attached to words (`chair.`), while ScanRefer's references are tokenised
  (`chair .`). Without PTB the n-grams rarely match; the PTB tokenizer in the aligned scorer
  splits them. The captions are also templated: 281 distinct captions for 2068 objects, e.g.
  "this is a black chair. it is at the end of the table." 176 times.
