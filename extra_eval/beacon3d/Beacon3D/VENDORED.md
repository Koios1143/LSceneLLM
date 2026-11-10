Vendored copy of https://github.com/beacon-3d/Beacon3D at commit `bda1203` (Apache-2.0,
see `LICENSE`): the two evaluators, `utils.py`, and `data/` (annotations, metadata and
the GPT system prompt for ScanNet, 3RScan and MultiScan). Unmodified. The evaluators
read `data/<domain>/...` relative to their cwd, so `extra_eval/score.py` runs them
from this directory.
