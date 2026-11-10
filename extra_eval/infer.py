#!/usr/bin/env python3
"""LSceneLLM (released ``release.pth``) inference on Scan2Cap, ScanQA, SQA3D, Real-3DQA and
Beacon3D QA (ScanNet, 3RScan, MultiScan), one process per GPU (``--num_chunks`` /
``--chunk_idx``, whole scenes per chunk).

The repository releases only the XR-QA loader (``datasets/SceneVerseDataset.HD_Hm3dQADataset``);
the ScanNet loaders the checkpoint was trained with (LL3DA's data: ScanRefer / Nr3D captions,
ScanQA, ...) are not released. Inputs are therefore built the way ``HD_Hm3dQADataset`` builds
them, for a single scan instead of a set of HM3D regions:

* the scan's OpenScene features (``openscene_features.py --pcds``: the SceneVerse point cloud,
  OpenSeg-distilled MinkUNet18A, 768-d per point), all points moved into the unit sphere
  (``pc_norm``), 160k points drawn for the dense (hd) tokens and 40k for the sparse ones
  (``down_sample``). Seeded by the scene name (``crc32``), so every question of a scene sees the
  same points (upstream draws per item from an unseeded RNG);
* the system prompt + ``<scene>`` + ``<vp_placehold>`` + instruction, padded to 256 tokens, and
  the model's ``forward(..., eval=True)`` (greedy decoding) under ``torch.cuda.amp.autocast`` as
  ``tools/runner_Adaptive_LLM.validate``; the answer is decoded and whitespace-joined as there.

Tasks and prompts (``TASK_PROMPT`` in ``datasets/SceneVerseDataset.py``, the first entry, as the
XR-QA loader uses):

* ``scanqa`` (val), ``sqa3d_test``, ``real3dqa_<split>``, ``beacon3d_<domain>_qa``:
  ``TASK_PROMPT['scene_qa'][0]`` with the question (SQA3D-style tasks: ``<situation> <question>``)
  lower-cased as LL3DA's ScanQA loader does; no visual prompt (box / click masks 0). Up to
  ``--qa_max_len`` tokens (20, the hard-coded value of ``AdaptiveLLM.forward``).
* ``scan2cap``: one item per ScanRefer val object (``scene|object``, 2068). The object is given
  by a click prompt (``click_query``, mask 1) at the centre of the Mask3D proposal (LEO's
  released ``mask.npz``, first 50 per scene, on the same SceneVerse points) with the highest
  box IoU to the ground-truth object (its SceneVerse instance points), in the normalised
  coordinates; ``TASK_PROMPT['object_caption'][0]``; up to ``--cap_max_len`` tokens. Each row
  records the IoU, so one run scores both @0.25 and @0.5.

Decoding: default = the repo's ``tools/generation_utils.generation`` (greedy, the whole sequence
re-run per token); ``--fast`` = ``fast_flex.kv_greedy`` (KV cache, same tokens; see there).

Writes ``<output_dir>/<task>/predictions_chunk_<i>.json`` (resumes per scene).
"""

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "extra_eval"))
os.chdir(REPO)  # the model reads ckpts/... relative to the repo
for k, v in {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
    os.environ.setdefault(k, v)

REAL3DQA_SPLITS = ["debiased_test", "rotation_0", "rotation_90", "rotation_180", "rotation_270"]
QA_SUFFIX = " Answer the question using a single word or phrase."
SQA_TYPE = {"what": 0, "is": 1, "how": 2, "can": 3, "which": 4}
DOMAIN = {"beacon3d_3rscan_qa": "3rscan", "beacon3d_multiscan_qa": "multiscan"}


# ---------------------------------------------------------------- annotations
def load_task(task, args):
    """-> list of {qid, scene_id, question | obj_id, answers, ...}."""
    A = args.anno_dir
    if task == "scan2cap":
        rows, seen = [], set()
        for r in json.load(open(os.path.join(A, "ScanRefer", "ScanRefer_filtered_val.json"))):
            key = f'{r["scene_id"]}|{r["object_id"]}'
            if key not in seen:
                seen.add(key)
                rows.append({"qid": key, "scene_id": r["scene_id"], "obj_id": int(r["object_id"]),
                             "object_name": r["object_name"]})
        return rows
    if task == "scanqa":
        rows = json.load(open(os.path.join(A, "ScanQA", "ScanQA_v1.0_val.json")))
        return [{"qid": r["question_id"], "scene_id": r["scene_id"], "question": r["question"],
                 "answers": r["answers"]} for r in rows]
    if task == "sqa3d_test":  # Chat-Scene's sqa3d_test.json: prompt = situation question + suffix
        out = []
        for i, r in enumerate(json.load(open(os.path.join(A, "sqa3d_test.json")))):
            assert r["prompt"].endswith(QA_SUFFIX), r["prompt"]
            out.append({"qid": i, "scene_id": r["scene_id"], "question": r["prompt"][:-len(QA_SUFFIX)],
                        "answers": r["ref_captions"], "sqa_type": r["sqa_type"]})
        return out
    if task.startswith("real3dqa_"):
        out = []
        for line in open(os.path.join(A, "real3dqa", task[len("real3dqa_"):] + ".jsonl")):
            x = json.loads(line)
            q = x["question"].strip()
            out.append({"qid": int(x["question_id"]), "scene_id": x["scene_id"],
                        "question": f'{x["situation"].strip()} {q}', "answers": [x["answer"]],
                        "sqa_type": SQA_TYPE.get(q.split(" ")[0].lower(), 5),
                        "question_type": x["question_type"]})
        return out
    if task.startswith("beacon3d_") and task.endswith("_qa"):
        domain = task[len("beacon3d_"):-len("_qa")]
        rows = json.load(open(os.path.join(args.beacon3d_dir, "data", domain, f"{domain}_qa.json")))
        return [{"qid": i, "scene_id": r["scene_id"], "question": r["question"], "answers": r["answers"],
                 "question_id": r["question_id"]} for i, r in enumerate(rows)]
    raise ValueError(task)


def domain_of(task):
    return DOMAIN.get(task, "scannet")


# ---------------------------------------------------------------- inputs
def pc_norm_params(pc):
    """``SceneVerseDataset.pc_norm`` split into (centroid, scale)."""
    centroid = np.mean(pc, axis=0)
    return centroid, np.max(np.sqrt(np.sum((pc - centroid) ** 2, axis=1)))


def down_sample(points, colors, instance_labels=None, featrues=None, npoint=None):
    """``SceneVerseDataset.down_sample``."""
    pcd_idxs = np.random.choice(len(points), size=npoint, replace=len(points) < npoint)
    return (points[pcd_idxs], colors[pcd_idxs],
            instance_labels[pcd_idxs] if instance_labels is not None else None,
            featrues[pcd_idxs] if featrues is not None else None)


def box_iou(a, b):
    """axis-aligned 3D IoU of (min, max) boxes"""
    inter = np.prod(np.clip(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]), 0, None))
    va, vb = np.prod(a[1] - a[0]), np.prod(b[1] - b[0])
    return float(inter / (va + vb - inter)) if inter > 0 else 0.0


class Scenes:
    def __init__(self, args):
        self.args = args

    def load(self, domain, scene_id):
        """-> dict of the scene's model inputs (shared by its items) + normalisation."""
        f = torch.load(os.path.join(self.args.features_dir, domain, f"{scene_id}.pth"), map_location="cpu",
                       weights_only=False)
        points = f["points"].numpy().astype(np.float32)
        features = f["features"].numpy().astype(np.float32)
        colors = f["colors"].numpy()
        labels = f["instance_labels"].numpy()
        centroid, scale = pc_norm_params(points)
        pts = (points - centroid) / scale
        np.random.seed(zlib.crc32(scene_id.encode()))
        hd_points, hd_features, _, hd_labels = down_sample(pts, features, colors, labels, npoint=160000)
        p40, _, _, f40 = down_sample(pts, colors, labels, features, npoint=self.args.n_points)
        return {"points": p40, "features": f40, "hd_points": hd_points, "hd_features": hd_features,
                "centroid": centroid, "scale": scale, "raw_points": points, "labels": labels}

    def scan2cap_click(self, sc, scene_id, obj_id):
        """best-IoU Mask3D proposal of the object -> (click xyz in normalised coords, iou, proposal index)"""
        import scipy.sparse as sp
        if "masks" not in sc:
            m = np.asarray(sp.load_npz(os.path.join(self.args.mask_dir, f"{scene_id}.mask.npz")).todense())[:50]
            assert m.shape[1] == len(sc["raw_points"]), (m.shape, len(sc["raw_points"]))
            sc["masks"] = m.astype(bool)
        gt = sc["raw_points"][sc["labels"] == obj_id]
        best = (0.0, -1)
        if len(gt):
            gbox = (gt.min(0), gt.max(0))
            for i, mk in enumerate(sc["masks"]):
                if mk.sum() == 0:
                    continue
                p = sc["raw_points"][mk]
                iou = box_iou((p.min(0), p.max(0)), gbox)
                if iou > best[0]:
                    best = (iou, i)
        if best[1] < 0:  # no overlapping proposal: click at the first non-empty proposal (scored 0 anyway)
            best = (0.0, int(np.argmax(sc["masks"].sum(1) > 0)))
        p = sc["raw_points"][sc["masks"][best[1]]]
        centre = (p.min(0) + p.max(0)) / 2
        return (centre - sc["centroid"]) / sc["scale"], best[0], best[1]


# ---------------------------------------------------------------- model
def build_model(args):
    from transformers import AutoConfig
    from utils.config import cfg_from_yaml_file
    from models.adaptive_llm import AdaptiveLLM
    cfg = cfg_from_yaml_file("cfgs/XR-QA-Evaluation.yaml")
    llm_config = AutoConfig.from_pretrained("ckpts/Llama-2-7b-hf/config.json")
    llm_config._attn_implementation = "eager"  # the flex attention exists only in LlamaAttention
    ns = SimpleNamespace(log_name="lscenellm_extra_eval", local_rank=0)
    model = AdaptiveLLM(llm_config, cfg.model, finetune=True, logger=None, args=ns)
    model.load_model_from_ckpt(args.ckpt, ns, finetune=True)
    return model.cuda().eval(), cfg


def make_batch(model, sc, instruction_text, click=None):
    from datasets.SceneVerseDataset import SYSTEM_PROMPT
    tok = model.tokenizer_ds
    text = "{} {} {} {}".format(SYSTEM_PROMPT, "<scene><scene_placehold></scene>", "<vp_placehold>", instruction_text)
    enc = tok.batch_encode_plus([text], max_length=256, padding="max_length", truncation="longest_first",
                                return_tensors="np")
    t = lambda x, dt=torch.float32: torch.from_numpy(np.asarray(x)).to(dt)[None].cuda()  # noqa: E731
    return {
        "points": t(sc["points"]), "features": t(sc["features"]),
        "hd_points": t(sc["hd_points"]), "hd_features": t(sc["hd_features"]),
        "level": ["scene"],
        "click_query": t(np.zeros((1, 3)) if click is None else np.asarray(click)[None]),
        "click_mask": t(np.zeros((1,)) if click is None else np.ones((1,))),
        "box_mask": t(np.zeros((1,))), "box_query": t(np.zeros((sc["features"].shape[-1],))),
        "instruction": t(enc["input_ids"][0], torch.int64),
        "instruction_mask": t(enc["attention_mask"][0]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True, help="space-separated")
    ap.add_argument("--ckpt", default="ckpts/release.pth")
    ap.add_argument("--features_dir", required=True, help="openscene_features.py outputs: <dir>/<domain>/<scan>.pth")
    ap.add_argument("--anno_dir", required=True, help="ScanRefer/, ScanQA/, sqa3d_test.json, real3dqa/")
    ap.add_argument("--beacon3d_dir", default="extra_eval/beacon3d/Beacon3D")
    ap.add_argument("--mask_dir", help="Mask3D proposals <scene>.mask.npz (scan2cap)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_idx", type=int, default=0)
    ap.add_argument("--n_points", type=int, default=40000)
    ap.add_argument("--qa_max_len", type=int, default=20)
    ap.add_argument("--cap_max_len", type=int, default=48)
    ap.add_argument("--qa_prompt", type=int, default=0, help="index into TASK_PROMPT['scene_qa'] (0 or 1)")
    ap.add_argument("--no_lower", action="store_true", help="keep the question's case")
    ap.add_argument("--fast", action="store_true", help="KV-cached greedy decoding (fast_flex.py)")
    ap.add_argument("--limit", type=int, default=0, help="first N items per task (checks)")
    ap.add_argument("--no_autocast", action="store_true", help="fp32 instead of the upstream fp16 autocast")
    ap.add_argument("--print_scenes", action="store_true", help="print '<domain> <scene>' per needed scan, exit")
    args = ap.parse_args()
    if args.print_scenes:
        need = sorted({(domain_of(t), r["scene_id"]) for t in args.tasks.split() for r in load_task(t, args)})
        print("\n".join(f"{d} {s}" for d, s in need))
        return

    import models.adaptive_llm as al
    from datasets.SceneVerseDataset import TASK_PROMPT
    from transformers import AutoTokenizer
    if args.fast:
        import fast_flex
        fast_flex.install()
        gen_impl = fast_flex.kv_greedy
    else:
        gen_impl = al.generation
    cur = {"max_len": args.qa_max_len}
    al.generation = lambda llm, **kw: gen_impl(llm, **{**kw, "max_length": cur["max_len"]})  # noqa: E731

    model, _ = build_model(args)
    tok = AutoTokenizer.from_pretrained("ckpts/Llama-2-7b-hf", add_bos_token=False)  # HD_Hm3dQADataset's
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": ['<vp_placehold>', '<scene>', '<scene_placehold>',
                                                          '</scene>', '<obj>', '</obj>']})
    model.tokenizer_ds = tok
    scenes = Scenes(args)
    stats = {}
    for task in args.tasks.split():
        items = load_task(task, args)
        if args.limit:
            items = items[:args.limit]
        scene_ids = sorted({r["scene_id"] for r in items})[args.chunk_idx::args.num_chunks]
        out_dir = os.path.join(args.output_dir, task)
        os.makedirs(out_dir, exist_ok=True)
        out_f = os.path.join(out_dir, f"predictions_chunk_{args.chunk_idx}.json")
        done = json.load(open(out_f)) if os.path.exists(out_f) else []
        done_scenes = {r["scene_id"] for r in done}
        t0, n = time.time(), 0
        for s_i, scene_id in enumerate(scene_ids):
            if scene_id in done_scenes:
                continue
            sc = scenes.load(domain_of(task), scene_id)
            rows = []
            for r in (x for x in items if x["scene_id"] == scene_id):
                click = None
                if task == "scan2cap":
                    click, iou, prop = scenes.scan2cap_click(sc, scene_id, r["obj_id"])
                    instr = TASK_PROMPT["object_caption"][0]["instruction"]
                    cur["max_len"] = args.cap_max_len
                else:
                    q = r["question"] if args.no_lower else r["question"].lower()
                    instr = TASK_PROMPT["scene_qa"][args.qa_prompt]["instruction"].format(locations="", question=q)
                    cur["max_len"] = args.qa_max_len
                batch = make_batch(model, sc, instr, click)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=not args.no_autocast):
                    output_ids, _, _ = model(batch, eval=True)
                ans = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)[0]
                ans = " ".join(filter(lambda w: w, ans.split(" ")))
                row = {**r, "pred": ans}
                if task == "scan2cap":
                    row.update({"iou": iou, "proposal": prop})
                rows.append(row)
                n += 1
            done.extend(rows)
            tmp = out_f + ".tmp"
            json.dump(done, open(tmp, "w"), indent=1)
            os.replace(tmp, out_f)
            el = time.time() - t0
            print(f"[{task}] scene {s_i + 1}/{len(scene_ids)} {scene_id}: {len(rows)} items, "
                  f"{el / max(n, 1):.2f} s/item, e.g. {rows[0]['pred']!r}", flush=True)
        stats[task] = {"items": n, "seconds": time.time() - t0}
    json.dump(stats, open(os.path.join(args.output_dir, f"stats_chunk_{args.chunk_idx}.json"), "w"), indent=1)
    print("INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
