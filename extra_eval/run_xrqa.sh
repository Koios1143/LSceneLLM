#!/bin/bash
# XR-QA val with the released LSceneLLM checkpoint, the upstream way (scripts/eval.sh:
# torch.distributed.launch main_ALLM.py --launcher pytorch --config cfgs/XR-QA-Evaluation.yaml
# --ckpts ckpts/release.pth --test) under torchrun (the code reads RANK and wraps the LLM in FSDP
# under --launcher pytorch), one item per process (extra_eval/torchrun_main.py explains why).
# NPROC processes / GPUs (default 1).
#   bash extra_eval/run_xrqa.sh [extra main_ALLM.py args]
# Steps (each skipped when its output exists):
#   1. OpenScene features of the val regions (extra_eval/openscene_features.py) with MINK_ENV,
#      a python env with MinkowskiEngine;
#   2. inference + LSceneLLM's own scoring (tools/runner_Adaptive_LLM.py validate), written to
#      experiments/XR-QA-Evaluation/cfgs/test_<EXP_NAME>/hd_scene_qa_-1_qa_*.json.
# Expected layout (repo-relative, the paths the upstream code uses; make data/ and ckpts/
# point into deps/, see env.sh.example):
#   data/SceneVerse/HM3D/annotations/qa/{train,val}.json   HF dataset Hoyard/XR-Scene XR-QA/
#   data/SceneVerse/HM3D/scan_data/{pcd_with_global_alignment,instance_id_to_label,room_center}
#   ckpts/release.pth (HF Hoyard/LSceneLLM), ckpts/Llama-2-7b-hf, ckpts/bert-base-uncased,
#   ckpts/scannet_openseg.pth.tar (OpenScene)
#   utils/capeval/meteor/data/paraphrase-en.gz (METEOR 1.5 paraphrase table, not in the repo)
# METEOR needs java on PATH (JAVA_HOME in env.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${LSCENELLM_ENV:-.venv}/bin/python"
MINK_PY="${MINK_ENV:?set MINK_ENV (python env with MinkowskiEngine)}/bin/python"
EXP_NAME="${EXP_NAME:-release_xrqa}"
[ -n "${JAVA_HOME:-}" ] && export PATH="$JAVA_HOME/bin:$PATH"

$MINK_PY extra_eval/openscene_features.py --anno data/SceneVerse/HM3D/annotations/qa/val.json

# the code torch.loads SceneVerse .pth files holding numpy arrays (torch >= 2.6 defaults to weights_only)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
OUT="experiments/XR-QA-Evaluation/cfgs/test_$EXP_NAME"
if [ ! -f "$OUT/hd_scene_qa_-1_qa_scores.json" ]; then
    $PY -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-1}" extra_eval/torchrun_main.py \
        --launcher pytorch --config cfgs/XR-QA-Evaluation.yaml --exp_name "$EXP_NAME" \
        --ckpts ckpts/release.pth --test "$@"
fi
cat "$OUT/hd_scene_qa_-1_qa_scores.json"; echo
