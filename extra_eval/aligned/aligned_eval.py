"""Aligned scoring: one text normalisation for every baseline (and just3d).

Decided with the user in the 2026-11 scoring audit. Every repo vendors an identical copy of
this file (``extra_eval/aligned/aligned_eval.py``); the per-repo ``score_aligned.py`` only
loads that repo's saved predictions, strips the model's own special tokens (no truncation)
and calls the functions here.

Modes (applied symmetrically to prediction and references):

* ``raw``        strings as-is (whitespace collapsed); LSceneLLM's XR protocol.
* ``ptb``        PTB tokenizer (lower-case, punctuation split off and dropped).
* ``clean_ptb``  ``clean_answer`` then PTB.  QA-style text (ScanQA, XR-QA).
* ``s2c``        ``sos <text> eos`` then PTB.  Scan2Cap (references come with sos/eos).

Exact match: ``clean_answer`` on both sides, then ``answer_match`` that first looks for an
exact hit over *all* references and only then for a substring hit (the official
Real-3DQA/Beacon3D matcher returns at the first substring hit and can miss a later exact
one).  An empty cleaned prediction scores 0 / 0.

Caption metrics come from pycocoevalcap 1.2 (BLEU-1..4, METEOR, ROUGE-L, CIDEr, x100);
PTB and METEOR need ``java`` on PATH.

SPICE (``spice_scores``, XR-SceneCaption) follows just3d's ``scripts/evaluate/spice.py``
exactly: whitespace collapsed (``strip_ws``), sentences split to <= 40 words
(``split_long_captions``), SPICE 1.0 + CoreNLP 3.6.0 run on JDK 17 with Nashorn on the
classpath; the score is the mean per-item All-F x100. An empty prediction scores 0.
"""

from __future__ import annotations

import re
from collections import OrderedDict

__all__ = ["clean_answer", "strip_ws", "answer_match", "em_scores", "normalise",
           "caption_metrics", "score_captions", "score_qa", "split_long_captions",
           "spice_scores"]


def clean_answer(data):
    """LEO / SQA3D / Real-3DQA ``clean_answer`` (same rules in every repo)."""
    if isinstance(data, list):
        return [clean_answer(x) for x in data]
    data = data.lower()
    data = re.sub(r"[ ]+$", "", data)
    data = re.sub(r"^[ ]+", "", data)
    data = re.sub(r" {2,}", " ", data)
    data = re.sub(r"\.[ ]{2,}", ". ", data)
    data = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", data)
    data = re.sub(r"ç", "c", data)
    data = re.sub(r"’", "'", data)
    data = re.sub(r"\bletf\b", "left", data)
    data = re.sub(r"\blet\b", "left", data)
    data = re.sub(r"\btehre\b", "there", data)
    data = re.sub(r"\brigth\b", "right", data)
    data = re.sub(r"\brght\b", "right", data)
    data = re.sub(r"\bbehine\b", "behind", data)
    data = re.sub(r"\btv\b", "TV", data)
    data = re.sub(r"\bchai\b", "chair", data)
    data = re.sub(r"\bwasing\b", "washing", data)
    data = re.sub(r"\bwaslked\b", "walked", data)
    data = re.sub(r"\boclock\b", "o'clock", data)
    data = re.sub(r"\bo'[ ]+clock\b", "o'clock", data)
    for d, w in (("0", "zero"), ("none", "zero"), ("1", "one"), ("2", "two"), ("3", "three"),
                 ("4", "four"), ("5", "five"), ("6", "six"), ("7", "seven"), ("8", "eight"),
                 ("9", "nine"), ("10", "ten"), ("11", "eleven"), ("12", "twelve"),
                 ("13", "thirteen"), ("14", "fourteen"), ("15", "fifteen"), ("16", "sixteen"),
                 ("17", "seventeen"), ("18", "eighteen"), ("19", "nineteen"), ("20", "twenty"),
                 ("23", "twenty-three")):
        data = re.sub(r"\b%s\b" % d, w, data)
    data = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", data)
    data = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\bbackwards\b", "backward", data)
    return data


def strip_ws(text: str) -> str:
    """Collapse whitespace (newlines would desync METEOR's line protocol) and drop ``|``."""
    return " ".join((text or "").replace("|", " ").split())


# --------------------------------------------------------------------------- exact match
def answer_match(pred: str, gts: list[str]) -> tuple[int, int]:
    """(exact, relaxed) on already-cleaned strings; exact over all refs first."""
    if pred in gts:
        return 1, 1
    p = "".join(pred.split())
    for gt in gts:
        g = "".join(gt.split())
        if p in g or g in p:
            return 0, 1
    return 0, 0


def em_scores(preds: dict, refs: dict) -> dict:
    """EM / EM-R in percent over ``preds`` {k: str} and ``refs`` {k: [str]}."""
    em = emr = 0
    for k, p in preds.items():
        p = clean_answer(p or "").strip()
        if not p:
            continue
        e, r = answer_match(p, [g.strip() for g in clean_answer(list(refs[k]))])
        em += e
        emr += r
    n = max(len(preds), 1)
    return {"EM": 100.0 * em / n, "EM-R": 100.0 * emr / n}


# --------------------------------------------------------------------------- caption metrics
def _ptb(d: dict) -> dict:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    return PTBTokenizer().tokenize({k: [{"caption": c} for c in v] for k, v in d.items()})


def normalise(refs: dict, preds: dict, mode: str) -> tuple[dict, dict]:
    """-> (gts {k: [str]}, res {k: [str]}) after ``mode`` on both sides."""
    keys = list(preds)
    gts = OrderedDict((k, [strip_ws(r) for r in refs[k]]) for k in keys)
    res = OrderedDict((k, [strip_ws(preds[k])]) for k in keys)
    if mode == "raw":
        return gts, res
    if mode == "clean_ptb":
        gts = OrderedDict((k, clean_answer(v)) for k, v in gts.items())
        res = OrderedDict((k, clean_answer(v)) for k, v in res.items())
    elif mode == "s2c":
        def wrap(s):
            s = s.strip()
            if not s.startswith("sos "):
                s = "sos " + s if s != "sos" else s
            if not s.endswith(" eos") and s != "sos eos":
                s = s + " eos"
            return s
        gts = OrderedDict((k, [wrap(r) for r in v]) for k, v in gts.items())
        res = OrderedDict((k, [wrap(r) for r in v]) for k, v in res.items())
    elif mode != "ptb":
        raise ValueError(mode)
    return _ptb(gts), _ptb(res)


def caption_metrics(gts: dict, res: dict) -> dict:
    """pycocoevalcap BLEU-1..4 / METEOR / ROUGE-L / CIDEr, x100, on prepared dicts."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    # an empty hypothesis breaks METEOR's java protocol on some inputs; a lone token scores ~0
    res = OrderedDict((k, [v[0] if v and v[0].strip() else "<empty>"]) for k, v in res.items())
    b, _ = Bleu(4).compute_score(gts, res)
    m, _ = Meteor().compute_score(gts, res)
    r, _ = Rouge().compute_score(gts, res)
    c, _ = Cider().compute_score(gts, res)
    return {"CIDEr": 100 * c, "BLEU-1": 100 * b[0], "BLEU-4": 100 * b[3],
            "METEOR": 100 * m, "ROUGE-L": 100 * r}


def score_captions(refs: dict, preds: dict, modes=("ptb", "raw"), spice=None) -> dict:
    """{mode: metrics} for caption-style benchmarks.

    Scene captions (``ptb`` in ``modes``, i.e. XR-SceneCaption) also get a ``spice`` block
    when ``ALIGNED_SPICE_DIR`` is set (``ALIGNED_SPICE_JAVA`` = JDK 17 java,
    ``ALIGNED_SPICE_CACHE``, ``ALIGNED_SPICE_SHARDS``, ``ALIGNED_SPICE_HEAP``).
    """
    import os
    out = {mode: caption_metrics(*normalise(refs, preds, mode)) for mode in modes}
    if spice is None:
        spice = "ptb" in modes and bool(os.environ.get("ALIGNED_SPICE_DIR"))
    if spice:
        out["spice"] = spice_scores(
            refs, preds, os.environ["ALIGNED_SPICE_DIR"],
            java=os.environ.get("ALIGNED_SPICE_JAVA", "java"),
            cache_dir=os.environ.get("ALIGNED_SPICE_CACHE", ""),
            shards=int(os.environ.get("ALIGNED_SPICE_SHARDS", "4")),
            heap=os.environ.get("ALIGNED_SPICE_HEAP", "8G"))
    return out


def score_qa(refs: dict, preds: dict, modes=("clean_ptb", "raw")) -> dict:
    """{mode: metrics + EM/EM-R} for QA-style benchmarks (EM is always clean_answer based)."""
    em = em_scores(preds, refs)
    return {mode: {**caption_metrics(*normalise(refs, preds, mode)), **em} for mode in modes}


# --------------------------------------------------------------------------- SPICE
# Vendored from just3d scripts/evaluate/just3d_eval/metrics.py (split_long_captions,
# compute_spice) so every baseline and just3d run the same SPICE.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_STRONG_RE = re.compile(r"(?<=[;:])\s+")
_CLAUSE_COORD_RE = re.compile(
    r",\s+(?=(?:and|but|or|nor|yet|so|while|whereas|whilst|which|who|where|then|"
    r"although|though|because|since)\b)",
    re.IGNORECASE,
)
_CLAUSE_COMMA_RE = re.compile(r",\s+")
_TERMINAL = ".!?"
_JDK_OPENS = [f"--add-opens=java.base/{pkg}=ALL-UNNAMED" for pkg in (
    "java.lang", "java.lang.reflect", "java.util", "java.util.concurrent", "java.io",
    "java.math", "java.net", "java.nio", "java.text")] + ["--add-opens=java.sql/java.sql=ALL-UNNAMED"]


def _emit(chunk: str) -> str:
    chunk = chunk.strip().rstrip(",;: ")
    if not chunk:
        return ""
    return chunk if chunk[-1] in _TERMINAL else chunk + "."


def _pack(pieces: list, max_words: int) -> list:
    out, cur, n = [], [], 0
    for piece in pieces:
        w = len(piece.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(piece)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


_TOK_RE = re.compile(r"\w+|[^\w\s]")


def _cap_tokens(sentence: str, max_words: int, max_tokens: int) -> list:
    """Whitespace-free run-ons ("6.2/12.27/2.02-...") count as one word but hundreds of
    CoreNLP tokens and exhaust the parser's heap; re-chunk only those by token."""
    toks = _TOK_RE.findall(sentence)
    if len(toks) <= max_tokens:
        return [sentence]
    return [" ".join(toks[i:i + max_words]) for i in range(0, len(toks), max_words)]


_UNSAFE_END_RE = re.compile(r"(?:^|[^\w])[^\W\d_]\.$")  # "... t a." : CoreNLP reads "a." as an initial


def _guard_groups(sentences: list, max_words: int, max_tokens: int) -> list:
    """CoreNLP keeps single-letter "a." as an initial and does not break there, so runs of
    sentences without a reliable boundary can merge into one huge parse; re-chunk such a
    run by token when it exceeds ``max_tokens``, ending each chunk with a standalone "."."""
    out, group = [], []

    def flush():
        toks = _TOK_RE.findall(" ".join(group))
        if len(toks) > max_tokens:
            out.extend(" ".join(toks[i:i + max_words]) + " ." for i in range(0, len(toks), max_words))
        else:
            out.extend(group)
        group.clear()

    for s in sentences:
        group.append(s)
        if not _UNSAFE_END_RE.search(s):
            flush()
    flush()
    return out


def split_long_captions(text: str, max_words: int = 40, max_tokens: int = 120) -> str:
    """Sentence breaks so no sentence exceeds ``max_words`` (CoreNLP's parser blows up)."""
    if not text:
        return ""
    out = []
    for sentence in _SENT_SPLIT_RE.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence.split()) <= max_words:
            for part in _cap_tokens(sentence, max_words, max_tokens):
                emitted = _emit(part)
                if emitted:
                    out.append(emitted)
            continue
        pieces = [sentence]
        for pattern in (_CLAUSE_STRONG_RE, _CLAUSE_COORD_RE, _CLAUSE_COMMA_RE):
            if all(len(p.split()) <= max_words for p in pieces):
                break
            pieces = [sub for p in pieces
                      for sub in (pattern.split(p) if len(p.split()) > max_words else [p])]
        hard = []
        for piece in pieces:
            words = piece.split()
            if len(words) <= max_words:
                hard.append(piece)
                continue
            hard += [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
        for chunk in _pack(hard, max_words):
            for part in _cap_tokens(chunk, max_words, max_tokens):
                emitted = _emit(part)
                if emitted:
                    out.append(emitted)
    return " ".join(_guard_groups(out, max_words, max_tokens))


def spice_scores(refs: dict, preds: dict, spice_dir: str, java: str = "java",
                 cache_dir: str = "", heap: str = "8G", shards: int = 4,
                 max_words: int = 40, timeout: int = 0) -> dict:
    """SPICE (All-F x100, mean over items; empty prediction = 0).

    ``spice_dir`` holds ``spice-1.0.jar`` and ``lib/`` (CoreNLP 3.6.0 jars + models,
    nashorn-core 15.4 + ASM 9.6); ``java`` is a JDK 17 binary.
    """
    import json
    import os
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    spice_dir = os.path.abspath(spice_dir)
    jar = os.path.join(spice_dir, "spice-1.0.jar")
    lib = os.path.join(spice_dir, "lib")
    if not os.path.exists(jar) or not any(f.startswith("nashorn-core") for f in os.listdir(lib)):
        raise RuntimeError(f"SPICE jar / nashorn jars missing under {spice_dir}")
    cache_dir = os.path.abspath(cache_dir) if cache_dir else ""
    if os.sep in java:  # java runs with cwd=spice_dir, so a relative path would not resolve
        java = os.path.abspath(java)

    def prep(t):
        return split_long_captions(strip_ws(t), max_words)

    payload, n_empty = [], 0
    for k in preds:
        rs = [c for c in (prep(r) for r in refs[k]) if c]
        p = prep(preds[k])
        if not rs:
            raise ValueError(f"item {k} has no references")
        if not p:
            n_empty += 1
            continue
        payload.append({"image_id": str(k), "test": p, "refs": rs})

    def batch(items, shard_cache):
        tmp = os.path.join(cache_dir or tempfile.gettempdir(), "io")
        os.makedirs(tmp, exist_ok=True)
        f = tempfile.NamedTemporaryFile("w+", dir=tmp, suffix=".json", delete=False)
        json.dump(items, f)
        f.close()
        out_path = f.name + ".out"
        cmd = [java, *_JDK_OPENS, f"-Xmx{heap}", "-Xss8m", "-cp", f"{jar}:{lib}/*",
               "edu.anu.spice.SpiceScorer", f.name, "-out", out_path, "-subset", "-silent"]
        if shard_cache:
            os.makedirs(shard_cache, exist_ok=True)
            cmd += ["-cache", shard_cache]
        try:
            subprocess.run(cmd, cwd=spice_dir, check=True, capture_output=True,
                           timeout=timeout if timeout > 0 else None)
            with open(out_path) as fh:
                return {it["image_id"]: it["scores"] for it in json.load(fh)}
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"SPICE failed (exit {exc.returncode}):\n"
                               f"{exc.stderr.decode()[-2000:]}") from exc
        finally:
            for path in (f.name, out_path):
                if os.path.exists(path):
                    os.remove(path)

    by_id = {}
    if payload:
        n = max(1, min(shards, len(payload)))
        bounds = [round(i * len(payload) / n) for i in range(n + 1)]
        chunks = [payload[bounds[i]:bounds[i + 1]] for i in range(n)]
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(batch, c, os.path.join(cache_dir, f"shard{i}") if cache_dir else "")
                    for i, c in enumerate(chunks) if c]
            for fu in futs:
                by_id.update(fu.result())

    def f_of(v):
        try:
            x = float(v["f"])
            return x if x == x else 0.0
        except (TypeError, ValueError, KeyError):
            return 0.0

    per = [f_of(by_id[it["image_id"]]["All"]) for it in payload] + [0.0] * n_empty
    out = {"SPICE": 100.0 * sum(per) / max(len(per), 1), "n": len(per), "n_empty_pred": n_empty}
    for cat in ("Object", "Attribute", "Relation", "Color", "Size", "Cardinality"):
        vals = [float(by_id[it["image_id"]][cat]["f"]) for it in payload
                if isinstance(by_id[it["image_id"]].get(cat), dict)
                and by_id[it["image_id"]][cat].get("f") not in (None, "NaN")
                and float(by_id[it["image_id"]][cat]["f"]) == float(by_id[it["image_id"]][cat]["f"])]
        out[f"SPICE_{cat}"] = 100.0 * sum(vals) / len(vals) if vals else None
    return out
