#!/bin/bash
# LSceneLLM inference env as a uv venv, ported to current GPUs: the README targets python 3.7 /
# old torch, which has no kernels for Blackwell (sm_120), so this uses python 3.10,
# torch 2.7.1+cu128 and transformers 4.40.2 (models/modeling_llama.py is a copy of the
# transformers 4.38-4.40 Llama), and builds the CUDA extensions the code imports:
# pointnet2_ops (erikwijmans/Pointnet2_PyTorch) and extensions/chamfer_dist; KNN_CUDA, no longer
# online, is replaced by an exact torch drop-in (extra_eval/knn_cuda_torch, used by dvae.py only).
#   bash extra_eval/build_env.sh
# Run on a GPU node with internet access (compiles CUDA code). Paths come from env.sh:
#   LSCENELLM_ENV        where the venv goes                       (default .venv)
#   LSCENELLM_CUDA_HOME  CUDA 12.x toolkit, >= 12.8                 (default /usr/local/cuda)
#   TORCH_CUDA_ARCH_LIST GPU archs to compile for                  (default: the visible GPU's)
#   UV_CACHE_DIR / TMPDIR   point these at a disk with room if $HOME is full
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
UV="${UV:-uv}"
ENV="${LSCENELLM_ENV:-.venv}"
export CUDA_HOME="${LSCENELLM_CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-16}"
$UV python install 3.10
[ -x "$ENV/bin/python" ] || $UV venv "$ENV" -p 3.10 --seed
PY="$ENV/bin/python"
PIP="$UV pip install --python $PY"
$PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$PIP transformers==4.40.2 "numpy<2" scipy h5py easydict termcolor einops pandas scikit-learn \
    "timm==0.6.13" tensorboardX tqdm pyyaml matplotlib wandb ninja sentencepiece protobuf \
    huggingface_hub
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$($PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}"
echo "CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# pointnet2_ops: its setup.py hard-codes TORCH_CUDA_ARCH_LIST (no sm_120), drop that line
if ! $PY -c "import torch, pointnet2_ops.pointnet2_utils" 2>/dev/null; then
    git clone --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch "$BUILD/p2"
    sed -i '/os.environ\["TORCH_CUDA_ARCH_LIST"\]/d' "$BUILD/p2/pointnet2_ops_lib/setup.py"
    $PIP --no-build-isolation "$BUILD/p2/pointnet2_ops_lib"
fi
# KNN_CUDA: github.com/unlimblue/KNN_CUDA and its wheel are gone; exact torch drop-in instead
$PY -c "import knn_cuda" 2>/dev/null || $PIP extra_eval/knn_cuda_torch
# chamfer distance (imported by models/dvae.py)
if ! $PY -c "import torch, chamfer" 2>/dev/null; then
    cp -r extensions/chamfer_dist "$BUILD/chamfer"
    $PIP --no-build-isolation "$BUILD/chamfer"
fi

$PY -c "
import torch, transformers, timm, chamfer
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('transformers', transformers.__version__, 'timm', timm.__version__)
x = torch.rand(1, 1024, 3, device='cuda')
print('furthest_point_sample ok', pointnet2_utils.furthest_point_sample(x, 16).shape)
print('knn ok', KNN(k=4, transpose_mode=True)(x, x[:, :8])[1].shape)
"
echo ENV_DONE
