"""Run main_ALLM.py under torchrun. The upstream launcher (torch.distributed.launch of old
torch) passed --local_rank=<i> to every process and main_ALLM.py asserts it equals the
process rank; torchrun only sets LOCAL_RANK, so this forwards it as the argument.

It also loads the LLM with eager attention: LSceneLLM's adaptive token selection
(``last_select_mask_list`` etc.) exists only in models/modeling_llama.py's ``LlamaAttention``,
while transformers >= 4.36 picks ``LlamaSdpaAttention`` by default, which rejects those arguments.

And it runs one item per process (config total_bs := number of processes): in evaluation
models/adaptive_llm.py stores the whole batch's dense (hd) scene tokens on every attention layer
but generates each item alone (batch 1), so modeling_llama.py's attention only works, and only
uses the right item's tokens, with one item per process. Generation is per item anyway, so this
changes neither the outputs nor the speed.

    python -m torch.distributed.run --nproc_per_node=N extra_eval/torchrun_main.py <main_ALLM.py args>
"""
import os
import runpy
import sys

from transformers import AutoConfig  # noqa: E402

_from_pretrained = AutoConfig.from_pretrained


def _eager_config(*args, **kwargs):
    cfg = _from_pretrained(*args, **kwargs)
    cfg._attn_implementation = "eager"
    return cfg


AutoConfig.from_pretrained = _eager_config

sys.path.insert(0, os.getcwd())
import utils.config  # noqa: E402

_cfg_from_yaml_file = utils.config.cfg_from_yaml_file


def _one_item_per_process(cfg_file):
    cfg = _cfg_from_yaml_file(cfg_file)
    cfg.total_bs = int(os.environ.get("WORLD_SIZE", "1"))
    return cfg


utils.config.cfg_from_yaml_file = _one_item_per_process
sys.argv = ["main_ALLM.py", f"--local_rank={os.environ.get('LOCAL_RANK', '0')}"] + sys.argv[1:]
runpy.run_path("main_ALLM.py", run_name="__main__")
