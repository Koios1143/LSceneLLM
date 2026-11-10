#!/bin/bash
# Score extra_eval/run_infer.sh outputs (extra_eval/score.py: own + aligned scoring). CPU only;
# METEOR / PTB need Java (JAVA_HOME), Beacon3D QA needs OPENAI_API_KEY.
#   bash extra_eval/score.sh
#   OUT         run_infer.sh output dir     (default eval_outputs/extra_eval)
#   BENCHES     comma list of scan2cap,scanqa,sqa3d_test,real3dqa,beacon3d   (default all)
#   QA_MODEL    Beacon3D QA judge           (default gpt-4o-mini-2024-07-18)
#   RESULT_DIR  where results go            (default extra_eval/results)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
[ -n "${JAVA_HOME:-}" ] && export PATH="$JAVA_HOME/bin:$PATH"
PY="${LSCENELLM_ENV:-.venv}/bin/python"
[ -n "${JUDGE_ENV:-}" ] && export JUDGE_PY="$(cd "$JUDGE_ENV" && pwd)/bin/python"   # python env with openai (Beacon3D judge)
$PY extra_eval/score.py --pred_dir "${OUT:-eval_outputs/extra_eval}" --result_dir "${RESULT_DIR:-extra_eval/results}" \
    --benches "${BENCHES:-scan2cap,scanqa,sqa3d_test,real3dqa,beacon3d}" --anno_dir "${EXTRA_ANNO:?}" \
    --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" || exit 1
echo SCORE_DONE
