"""OpenScene point features of the HM3D regions an XR split uses, as LSceneLLM expects them in
``data/SceneVerse/OpenScene_Scan_Features/HM3D/<scan>_<region>.pth``.

Same computation as ``models/openscene/run/preprocess.py`` (feature_type distill, config
scannet/ours_openseg_pretrained.yaml: MinkUNet18A, 768-d OpenSeg-distilled, checkpoint
scannet_openseg.pth.tar, voxel 2 cm, colours /127.5 - 1 as input, no augmentation), same
output dict (features fp16 per input point, points, colors, instance_labels, inds_reverse).
Differences: only the regions referenced by the given annotation files are processed (the
upstream script runs over every SceneVerse HM3D region), and the CLIP / SharedArray imports
that the distill path never uses are not needed.

    python extra_eval/openscene_features.py --anno data/SceneVerse/HM3D/annotations/qa/val.json

``--pcds`` instead processes the given SceneVerse-format point clouds (ScanNet / MultiScan
``pcd_with_global_alignment/<scan>.pth``, 3RScan ``<scan>/pcd-align.pth``; tuples whose first two
entries are xyz and rgb 0..255 and whose last is the instance labels), each saved as
``<out>/<scan>.pth``:

    python extra_eval/openscene_features.py --pcds .../scene0011_00.pth ... --out .../ScanNet
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models", "openscene"))
import collections, collections.abc  # noqa: E401,E402  voxelization_utils: `from collections import Sequence` (python < 3.10)
collections.Sequence = collections.abc.Sequence
from MinkowskiEngine import SparseTensor  # noqa: E402
from dataset.voxelizer import Voxelizer  # noqa: E402
from models.disnet import DisNet  # noqa: E402


class Cfg:
    feature_2d_extractor = "openseg"
    arch_3d = "MinkUNet18A"


def regions(anno_files):
    out = set()
    for f in anno_files:
        for scan, eps in json.load(open(f)).items():
            for e in eps:
                for q in e.get("qa_pairs", [e]):
                    for r in q["region_id"].split("-"):
                        out.add(f"{scan}_{r}")
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", nargs="+", help="XR annotation json(s) (scan -> episodes)")
    ap.add_argument("--pcds", nargs="+", help="SceneVerse-format point cloud files (instead of --anno)")
    ap.add_argument("--scan_data", default="data/SceneVerse/HM3D/scan_data/pcd_with_global_alignment")
    ap.add_argument("--ckpt", default="ckpts/scannet_openseg.pth.tar")
    ap.add_argument("--out", default="data/SceneVerse/OpenScene_Scan_Features/HM3D")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th region starting at i")
    args = ap.parse_args()

    torch.manual_seed(1463)
    np.random.seed(1463)
    model = DisNet(Cfg()).cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    torch.backends.cudnn.enabled = False
    vox = Voxelizer(voxel_size=0.02, clip_bound=None, use_augmentation=False)

    assert bool(args.anno) != bool(args.pcds), "give --anno or --pcds"
    if args.anno:
        todo = [(r, os.path.join(args.scan_data, f"{r}.pth")) for r in regions(args.anno)]
    else:  # 3RScan: <scan>/pcd-align.pth, otherwise <scan>.pth
        todo = sorted((os.path.basename(os.path.dirname(p)) if os.path.basename(p) == "pcd-align.pth"
                       else os.path.basename(p)[:-len(".pth")], p) for p in args.pcds)
    i, n = map(int, args.shard.split("/"))
    todo = todo[i::n]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(todo)} regions (shard {args.shard})", flush=True)
    for k, (name, src) in enumerate(todo):
        dst = os.path.join(args.out, f"{name}.pth")
        if os.path.exists(dst):
            continue
        pcd = torch.load(src, weights_only=False)
        locs_in, feats_in, labels_in = pcd[0], pcd[1], pcd[-1]
        if labels_in is None:
            labels_in = np.zeros_like(locs_in[:, 0])
        locs, feats, _, inds_reverse, _ = vox.voxelize(locs_in, feats_in, labels_in, return_ind=True)
        coords = torch.cat((torch.ones(len(locs), 1, dtype=torch.int), torch.from_numpy(locs).int()), 1)
        feats = torch.from_numpy(feats).float() / 127.5 - 1.0
        inds_reverse = torch.from_numpy(inds_reverse).long()
        with torch.no_grad():
            pred = model(SparseTensor(feats.cuda(), coords.cuda()))[inds_reverse.cuda()]
        tmp = dst + ".tmp"
        torch.save({"features": pred.half().cpu(), "points": torch.from_numpy(locs_in),
                    "colors": torch.from_numpy(feats_in), "instance_labels": torch.from_numpy(labels_in).long(),
                    "inds_reverse": inds_reverse}, tmp)
        os.replace(tmp, dst)
        print(f"[{k + 1}/{len(todo)}] {name}: {len(locs_in)} points, {len(locs)} voxels", flush=True)


if __name__ == "__main__":
    main()
