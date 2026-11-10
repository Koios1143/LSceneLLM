#!/usr/bin/env python3
"""Score ``infer.py`` outputs. CPU only; METEOR / PTB need Java, Beacon3D QA needs OPENAI_API_KEY.

Two scorings, both written:

* **own** (``results/<bench>/scores.json``): LSceneLLM's own scorer, ``utils/capeval`` BLEU /
  CIDEr / METEOR / ROUGE-L on the raw text, no PTB tokenizer, as ``runner_Adaptive_LLM.validate``
  (the same code as LL3DA's ``utils/capeval``). Scan2Cap as LL3DA's ``evaluate_densecap`` (the
  checkpoint's caption data is LL3DA's): references = ScanRefer val descriptions, tokens cut to
  30, wrapped in ``sos ... eos``; prediction ``sos <caption> eos``, ``sos eos`` when the object's
  best proposal IoU is not above the threshold (0.25 / 0.5). EM / EM-R (ScanQA, SQA3D) by
  Real-3DQA's official ``clean_answer`` / ``answer_match`` (empty -> ``<empty>``); SQA3D also per
  question type. Real-3DQA by the official ``evaluate_debiased_exact_match.py`` /
  ``evaluate_rotation_robustness.py`` (``real3dqa/``). Beacon3D QA by the official
  ``evaluate_qa.py`` (``beacon3d/Beacon3D``) with the GPT judge ``--qa_model``.
* **aligned** (``results/aligned/``): the cross-baseline scorer ``aligned/aligned_eval.py``
  (identical in every baseline): Scan2Cap mode ``s2c`` (sos/eos + PTB both sides, untruncated
  references ``deps/aligned/scan2cap_val_corpus.json``, unmatched -> ``""``), ScanQA
  ``clean_ptb`` and ``raw`` (+ EM), SQA3D EM / EM-R.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "real3dqa"))
REAL3DQA_SPLITS = ["debiased_test", "rotation_0", "rotation_90", "rotation_180", "rotation_270"]
EXPECTED = {"scanqa": 4675, "sqa3d_test": 3519, "real3dqa_debiased_test": 1485, "real3dqa_rotation_0": 750,
            "real3dqa_rotation_90": 750, "real3dqa_rotation_180": 750, "real3dqa_rotation_270": 750,
            "beacon3d_scannet_qa": 1155, "beacon3d_3rscan_qa": 816, "beacon3d_multiscan_qa": 477,
            "scan2cap": 2068}
EMPTY = "<empty>"
SQA_TYPES = ["what", "is", "how", "can", "which", "others"]
BEACON_TAGS = ["class", "appearance", "geometry", "spatial", "existence", "functionality"]


def load(pred_dir, task, allow_partial=False):
    rows = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, task, "predictions_chunk_*.json"))):
        for r in json.load(open(f)):
            rows.setdefault(r["qid"], r)
    if not rows:
        return None
    if not allow_partial:
        assert len(rows) == EXPECTED[task], f"{task}: {len(rows)} predictions, expected {EXPECTED[task]}"
    return rows


def capeval(corpus, candidates):
    """utils/capeval as runner_Adaptive_LLM.score_captions (+ BLEU-1)."""
    import utils.capeval.bleu.bleu as capbleu
    import utils.capeval.cider.cider as capcider
    import utils.capeval.meteor.meteor as capmeteor
    import utils.capeval.rouge.rouge as caprouge
    b = capbleu.Bleu(4).compute_score(corpus, candidates)[0]
    c = capcider.Cider().compute_score(corpus, candidates)[0]
    r = caprouge.Rouge().compute_score(corpus, candidates)[0]
    m = capmeteor.Meteor().compute_score(corpus, candidates)[0]
    return {"CIDEr": 100 * c, "BLEU-1": 100 * b[0], "BLEU-4": 100 * b[3], "METEOR": 100 * m, "ROUGE-L": 100 * r}


def scan2cap_corpus(anno_dir):
    """LL3DA eval_utils/evaluate_densecap.prepare_corpus: key scene|obj|name, sos + token[:30] + eos"""
    corpus = OrderedDict()
    for d in json.load(open(os.path.join(anno_dir, "ScanRefer", "ScanRefer_filtered_val.json"))):
        key = f'{d["scene_id"]}|{d["object_id"]}|{d["object_name"]}'
        corpus.setdefault(key, []).append(" ".join(["sos"] + d["token"][:30] + ["eos"]))
    return corpus


def score_scan2cap(rows, args):
    corpus = scan2cap_corpus(args.anno_dir)
    res = {}
    for thr in (0.25, 0.5):
        cands = OrderedDict()
        for key in corpus:
            r = rows[key.rsplit("|", 1)[0]]
            ok = r["iou"] > thr  # evaluate_densecap: match_box_ious > test_min_iou
            cands[key] = [("sos " + r["pred"] + " eos").replace("  ", " ") if ok else "sos eos"]
        res.update({f"{k}@{thr}": v for k, v in capeval(corpus, cands).items()})
        res[f"matched@{thr}"] = f"{sum(r['iou'] > thr for r in rows.values())}/{len(rows)}"
    return res


def em_scores(rows, by_type=False):
    import evaluate_debiased_exact_match as dem
    em, emr, per = [], [], {}
    for r in rows:
        pred = dem.clean_answer(r["pred"])
        pred = pred if pred.strip() else EMPTY
        e, er = dem.answer_match(pred, [dem.clean_answer(a) for a in r["answers"]])
        em.append(e)
        emr.append(er)
        if by_type:
            per.setdefault(SQA_TYPES[r["sqa_type"]], []).append((e, er))
    out = {"EM": 100 * float(np.mean(em)), "EM-R": 100 * float(np.mean(emr)),
           "empty_preds": sum(not dem.clean_answer(r["pred"]).strip() for r in rows)}
    for t in SQA_TYPES if by_type else []:
        if t in per:
            out[f"EM_{t}"] = 100 * float(np.mean([x[0] for x in per[t]]))
            out[f"EM-R_{t}"] = 100 * float(np.mean([x[1] for x in per[t]]))
    return out


def score_scanqa(rows):
    corpus = OrderedDict((str(r["qid"]), r["answers"]) for r in rows.values())
    cands = OrderedDict((str(r["qid"]), [r["pred"]]) for r in rows.values())
    return {**capeval(corpus, cands), **em_scores(rows.values())}


def score_real3dqa(pred_dir, out):
    import evaluate_debiased_exact_match as dem
    import evaluate_rotation_robustness as rot
    res, files = {}, {}
    for split in REAL3DQA_SPLITS:
        rows = load(pred_dir, f"real3dqa_{split}")
        if rows is None:
            return None
        files[split] = out / f"pred_{split}.json"
        json.dump([{"question_id": r["qid"], "scene_id": r["scene_id"], "response_gt": r["answers"],
                    "response_pred": r["pred"] if dem.clean_answer(r["pred"]).strip() else EMPTY,
                    "prompt": r["question"]} for r in rows.values()], open(files[split], "w"), indent=1)
        res[split] = dem.evaluate_em(str(files[split]))
        res[split]["empty_preds"] = sum(not dem.clean_answer(r["pred"]).strip() for r in rows.values())
    res["rotation"] = rot.quick_consistency_score([str(files[s]) for s in REAL3DQA_SPLITS[1:]])
    log = []
    for cmd in [[sys.executable, str(HERE / "real3dqa" / "evaluate_debiased_exact_match.py"),
                 str(files["debiased_test"]), "--name", "LSceneLLM"],
                [sys.executable, str(HERE / "real3dqa" / "evaluate_rotation_robustness.py")]
                + [str(files[s]) for s in REAL3DQA_SPLITS[1:]] + ["--name", "LSceneLLM"]]:
        log.append(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    (out / "official_scripts.txt").write_text("\n".join(log).replace(str(REPO) + "/", ""))
    return res


def beacon_metrics(pq):
    gpt = lambda xs: (float(np.mean(xs)) - 1) / 4 * 100  # noqa: E731
    sc = [it["score"] for it in pq]
    m = {"case": gpt(sc),
         "object": 100 * float(np.mean([all(x >= 4 for x in sc[i:i + 3]) for i in range(0, len(sc), 3)])),
         "EM": 100 * float(np.mean([it["em"] for it in pq])),
         "EM-R": 100 * float(np.mean([it["em_r"] for it in pq]))}
    for t in BEACON_TAGS:
        xs = [it["score"] for it in pq if it["tag"].lower() == t]
        m[t] = gpt(xs) if xs else None
    return m


def score_beacon3d(rows, domain, out, args):
    beacon = (HERE / "beacon3d" / "Beacon3D").resolve()
    src = json.load(open(beacon / "data" / domain / f"{domain}_qa.json"))
    infer = [{"question_id": s["question_id"], "scene_id": s["scene_id"], "question": s["question"],
              "pred": rows[i]["pred"], "answers": s["answers"]} for i, s in enumerate(src)]
    json.dump(infer, open(out / "infer_qa.json", "w"), indent=1)
    processed = out / "infer_qa_processed.json"
    processed.unlink(missing_ok=True)  # evaluate_qa.py silently reuses an existing one
    # the judge needs the openai package: JUDGE_PY = a python with openai (default: this one)
    cmd = [os.environ.get("JUDGE_PY") or sys.executable, "evaluate_qa.py", "--infer", str(out / "infer_qa.json"), "--domain", domain,
           "--model", args.qa_model, "--output", str(processed)]
    p = subprocess.run(cmd, cwd=beacon, capture_output=True, text=True)
    (out / "score_qa.txt").write_text(p.stdout.replace(str(REPO) + "/", ""))
    if p.returncode != 0:
        raise SystemExit(f"{domain}: evaluate_qa.py failed\n{p.stderr[-3000:]}")
    m = beacon_metrics(json.load(open(processed)))
    official = float(p.stdout.split("GPT-Score =")[1].split()[0])
    assert abs(official - m["case"]) < 0.01, (official, m)
    return {"qa_judge": args.qa_model, **m}


# ---------------------------------------------------------------- aligned
def aligned(pred_dir, result_dir, benches):
    sys.path.insert(0, str(REPO / "deps" / "pycocoevalcap-1.2"))
    sys.path.insert(0, str(HERE / "aligned"))
    import aligned_eval as ae
    out = result_dir / "aligned"
    out.mkdir(parents=True, exist_ok=True)
    sfile = out / "summary.json"
    summary = json.load(open(sfile)) if sfile.exists() else {}

    def write(bench, variant, payload):
        payload = {"bench": bench, "variant": variant, **payload}
        json.dump(payload, open(out / f"{bench}__{variant}.json", "w"), indent=1)
        summary[f"{bench}__{variant}"] = payload
        print(f"[aligned {bench}__{variant}]", json.dumps(payload["metrics"]), flush=True)

    if "scan2cap" in benches and (rows := load(pred_dir, "scan2cap")):
        corpus = json.load(open(REPO / "deps" / "aligned" / "scan2cap_val_corpus.json"))
        assert len(corpus) == 2068
        for thr in (0.25, 0.5):
            preds = OrderedDict((k, rows[k]["pred"].strip() if rows[k]["iou"] > thr else "") for k in sorted(corpus))
            write("scan2cap", f"iou{thr}", {"n": len(preds), "matched": sum(bool(v) for v in preds.values()),
                                            "metrics": ae.score_captions(corpus, preds, modes=("s2c",))})
    if "scanqa" in benches and (rows := load(pred_dir, "scanqa")):
        refs = OrderedDict((r["qid"], list(r["answers"])) for r in rows.values())
        preds = OrderedDict((r["qid"], r["pred"]) for r in rows.values())
        write("scanqa", "val", {"n": len(preds), "metrics": ae.score_qa(refs, preds, modes=("clean_ptb", "raw"))})
    if "sqa3d_test" in benches and (rows := load(pred_dir, "sqa3d_test")):
        refs = {r["qid"]: list(r["answers"]) for r in rows.values()}
        preds = {r["qid"]: r["pred"] for r in rows.values()}
        write("sqa3d", "test", {"n": len(preds), "metrics": {"em": ae.em_scores(preds, refs)}})
    json.dump(OrderedDict(sorted(summary.items())), open(sfile, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, help="infer.py --output_dir")
    ap.add_argument("--result_dir", default=str(HERE / "results"))
    ap.add_argument("--benches", default="scan2cap,scanqa,sqa3d_test,real3dqa,beacon3d")
    ap.add_argument("--anno_dir", default=os.environ.get("EXTRA_ANNO"))
    ap.add_argument("--qa_model", default="gpt-4o-mini-2024-07-18",
                    help="Beacon3D GPT judge (upstream default gpt-4o-2024-11-20)")
    args = ap.parse_args()
    pred_dir = os.path.abspath(args.pred_dir)
    result_dir = Path(args.result_dir).resolve()
    os.chdir(REPO)
    summary_file = result_dir / "summary.json"
    summary = json.load(open(summary_file)) if summary_file.exists() else {}
    benches = args.benches.split(",")
    for bench in benches:
        jobs = ([(f"beacon3d_{d}", f"beacon3d_{d}_qa") for d in ["scannet", "3rscan", "multiscan"]]
                if bench == "beacon3d" else [(bench, bench)])
        for name, task in jobs:
            out = result_dir / name
            out.mkdir(parents=True, exist_ok=True)
            if name == "real3dqa":
                res = score_real3dqa(pred_dir, out)
            else:
                rows = load(pred_dir, task)
                if rows is None:
                    res = None
                else:
                    json.dump(list(rows.values()), open(out / "preds.json", "w"), indent=1)
                    if name == "scan2cap":
                        res = score_scan2cap(rows, args)
                    elif name == "scanqa":
                        res = score_scanqa(rows)
                    elif name == "sqa3d_test":
                        res = em_scores(rows.values(), by_type=True)
                    else:
                        res = score_beacon3d(rows, name[len("beacon3d_"):], out, args)
            if res is None:
                print(f"[{name}] no predictions in {pred_dir}, skipped")
                continue
            res = {"pred_dir": os.path.relpath(pred_dir, REPO), **res}
            json.dump(res, open(out / "scores.json", "w"), indent=1)
            summary[name] = res
            print(f"[{name}]", json.dumps(res, indent=1), flush=True)
        if bench == "beacon3d" and all(n in summary for n, _ in jobs):
            pooled = [it for n, _ in jobs for it in json.load(open(result_dir / n / "infer_qa_processed.json"))]
            per = [summary[n] for n, _ in jobs]
            overall = {"qa_judge": args.qa_model, "pooled": beacon_metrics(pooled),
                       "mean": {k: float(np.mean([p[k] for p in per if p.get(k) is not None]))
                                for k in beacon_metrics(pooled)}}
            (result_dir / "beacon3d_overall").mkdir(parents=True, exist_ok=True)
            json.dump(overall, open(result_dir / "beacon3d_overall" / "scores.json", "w"), indent=1)
            summary["beacon3d_overall"] = overall
    json.dump(summary, open(summary_file, "w"), indent=1)
    aligned(pred_dir, result_dir, benches)


if __name__ == "__main__":
    main()
