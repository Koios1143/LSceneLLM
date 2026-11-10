#!/bin/bash
# LSceneLLM (released ckpts/release.pth) on Scan2Cap, ScanQA, SQA3D test, Real-3DQA and Beacon3D QA
# (extra_eval/infer.py), one process per visible GPU (whole scenes per process).
#   bash extra_eval/run_infer.sh                       # the repo's own greedy decoding
#   FAST=1 bash extra_eval/run_infer.sh                # KV-cached decoding (infer.py --fast), same tokens
#   TASKS="scanqa sqa3d_test" FAST=1 bash extra_eval/run_infer.sh
# Steps:
#   1. OpenScene features (extra_eval/openscene_features.py --pcds, MINK_ENV) of every scan the
#      tasks use -> $SCAN_FEATURES/<domain>/<scan>.pth (skipped when present);
#   2. inference -> $OUT/<task>/predictions_chunk_<i>.json (resumes per scene).
# Paths come from env.sh (see env.sh.example):
#   LSCENELLM_ENV, MINK_ENV
#   EXTRA_ANNO      ScanRefer/ScanRefer_filtered_val.json, ScanQA/ScanQA_v1.0_val.json, sqa3d_test.json, real3dqa/*.jsonl
#   SCANNET_PCD     SceneVerse ScanNet pcd_with_global_alignment/   (<scan>.pth)
#   RSCAN_PCD       SceneVerse 3RScan dir                           (<scan>/pcd-align.pth)
#   MULTISCAN_PCD   SceneVerse MultiScan pcd_with_global_alignment/ (<scan>.pth)
#   MASK3D_DIR      Mask3D proposals on the SceneVerse ScanNet points (<scan>.mask.npz, LEO's release)
#   SCAN_FEATURES   feature output dir                (default data/SceneVerse/OpenScene_Scan_Features)
#   OUT             output dir                        (default eval_outputs/extra_eval)
#   TASKS           space-separated                   (default: all below)
#   EXTRA_ARGS      more infer.py args (e.g. "--limit 20")
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${LSCENELLM_ENV:-.venv}/bin/python"
MINK_PY="${MINK_ENV:?set MINK_ENV}/bin/python"
OUT="${OUT:-eval_outputs/extra_eval}"
FEAT="${SCAN_FEATURES:-data/SceneVerse/OpenScene_Scan_Features}"
REAL3DQA="real3dqa_debiased_test real3dqa_rotation_0 real3dqa_rotation_90 real3dqa_rotation_180 real3dqa_rotation_270"
TASKS="${TASKS:-scan2cap scanqa sqa3d_test $REAL3DQA beacon3d_scannet_qa beacon3d_3rscan_qa beacon3d_multiscan_qa}"
COMMON=(--tasks "$TASKS" --features_dir "$FEAT" --anno_dir "${EXTRA_ANNO:?}" --mask_dir "${MASK3D_DIR:-}"
        --beacon3d_dir extra_eval/beacon3d/Beacon3D)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# 1. features of the scans the tasks need
declare -A SRC=([scannet]="${SCANNET_PCD:-}/%s.pth" [3rscan]="${RSCAN_PCD:-}/%s/pcd-align.pth"
                [multiscan]="${MULTISCAN_PCD:-}/%s.pth")
for d in scannet 3rscan multiscan; do
    mapfile -t PCDS < <($PY extra_eval/infer.py "${COMMON[@]}" --output_dir "$OUT" --print_scenes \
        | awk -v d="$d" '$1 == d {print $2}' | while read -r s; do
            [ -f "$FEAT/$d/$s.pth" ] || printf "${SRC[$d]}\n" "$s"; done)
    if [ ${#PCDS[@]} -gt 0 ]; then
        echo "features: $d ${#PCDS[@]} scans"
        $MINK_PY extra_eval/openscene_features.py --pcds "${PCDS[@]}" --out "$FEAT/$d" || exit 1
    fi
done

# 2. inference
gpu_list="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
FASTARG=(); [ "${FAST:-0}" = 1 ] && FASTARG=(--fast)
echo "out=$OUT tasks=[$TASKS] chunks=$CHUNKS fast=${FAST:-0}"
mkdir -p "$OUT"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} $PY extra_eval/infer.py "${COMMON[@]}" --output_dir "$OUT" \
        --num_chunks "$CHUNKS" --chunk_idx "$IDX" "${FASTARG[@]}" ${EXTRA_ARGS:-} \
        > "$OUT/infer_chunk_$IDX.log" 2>&1 &
done
wait
tail -n 2 "$OUT"/infer_chunk_*.log
grep -q INFER_DONE "$OUT/infer_chunk_0.log" && echo INFER_DONE
