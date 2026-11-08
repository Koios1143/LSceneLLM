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
