"""Keypoint LocSim evaluation on results derived from the mini partition ground truth.

Writes a minimal results file, i.e. one whose detections carry only the keys the evaluation needs
(`image_id`, `category_id`, `score`, `keypoints_3d`; no 2D `keypoints`), with the ground truth
people perturbed by 3D noise, missed people and false positives, and runs `coco_eval` with the
iou_types fifa15-3d-locsim, body25-3d-locsim and coco-3d-locsim on it.

    python tests/test_keypoint_locsim.py [--gt person_keypoints_mini.json] [--out results.json]
                                         [--sigma 0.1] [--miss 0.1] [--extra 0.05] [--tau 1]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from sskit.coco import coco_eval, LOCSIM_KEYPOINT_SUBSETS

GT = Path(os.environ.get("SSKIT_MINI_KEYPOINTS",
                         "/home/hakan/data/SoccerNet/SpiideoSynLoc/annotations/person_keypoints_mini.json"))
RESULT_KEYS = {"image_id", "category_id", "score", "keypoints_3d"}


def detection(image, kp3, score):
    return dict(image_id=image["id"], category_id=1, score=float(score),
                keypoints_3d=[round(float(x), 4) for x in kp3.ravel()])


def perturbed_results(gt, rng, sigma=0.1, miss=0.1, extra=0.05):
    """Detections from the ground truth: 3D keypoints with N(0, sigma) metres of noise, a fraction
    `miss` of the people dropped, and `extra` false positives per ground truth person (copies of
    random people moved elsewhere on the pitch)."""
    images = {im["id"]: im for im in gt["images"]}
    anns = gt["annotations"]
    res = []
    for a in anns:
        if rng.random() < miss:
            continue
        kp3 = np.array(a["keypoints_3d"], float).reshape(-1, 4)[:, :3]
        kp3 = kp3 + rng.normal(scale=sigma, size=kp3.shape)
        res.append(detection(images[a["image_id"]], kp3, rng.uniform(0.5, 1)))
    for _ in range(int(round(extra * len(anns)))):
        a = anns[rng.integers(len(anns))]
        kp3 = np.array(a["keypoints_3d"], float).reshape(-1, 4)[:, :3]
        kp3[:, :2] += rng.uniform(-20, 20, size=2)
        res.append(detection(images[a["image_id"]], kp3, rng.uniform(0, 0.6)))
    return res


def run(gt_path, out_path, sigma=0.1, miss=0.1, extra=0.05, tau=1, seed=0, log=print):
    """Write the perturbed minimal results file to `out_path` and evaluate it; returns
    {iou_type: metrics}."""
    with open(gt_path) as fd:
        gt = json.load(fd)
    res = perturbed_results(gt, np.random.default_rng(seed), sigma, miss, extra)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fd:
        json.dump(res, fd)
    log(f"wrote {out_path}: {len(res)} detections for {len(gt['annotations'])} people in {len(gt['images'])} images "
        f"(sigma={sigma} m, miss={miss}, extra={extra}, tau={tau} m)")
    return {typ: coco_eval(str(gt_path), res, typ, log, locsim_tau=tau) for typ in LOCSIM_KEYPOINT_SUBSETS}


def test_keypoint_locsim(tmp_path):
    import pytest
    if not GT.exists():
        pytest.skip(f"{GT} not found (set SSKIT_MINI_KEYPOINTS)")
    log = lambda s: None

    # unperturbed copy of the ground truth: perfect scores for every subset
    exact = run(GT, tmp_path / "exact.json", sigma=0, miss=0, extra=0, log=log)
    for typ, m in exact.items():
        assert m["AP"] == m["AP50"] == m["recall_50"] == m["precision_50"] == m["frame_accuracy"] == 1, (typ, m)

    # noisy, with misses and false positives: still matched well at LocSim 0.5, recall bounded by the misses
    noisy = run(GT, tmp_path / "noisy.json", sigma=0.1, miss=0.1, extra=0.05, log=log)
    with open(tmp_path / "noisy.json") as fd:
        res = json.load(fd)
    assert all(set(d) == RESULT_KEYS for d in res)
    for typ, m in noisy.items():
        assert m["AP50"] > 0.8 and 0.8 < m["recall_50"] <= 0.95, (typ, m)
        assert m["AP"] < exact[typ]["AP"], typ

    # a tighter tau lowers every similarity, hence the AP over the LocSim thresholds
    tight = run(GT, tmp_path / "tight.json", sigma=0.1, miss=0.1, extra=0.05, tau=0.3, log=log)
    for typ in tight:
        assert tight[typ]["AP"] < noisy[typ]["AP"], typ


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=GT, type=Path)
    ap.add_argument("--out", default="results_mini_locsim.json", type=Path)
    ap.add_argument("--sigma", default=0.1, type=float, help="3D keypoint noise in metres")
    ap.add_argument("--miss", default=0.1, type=float, help="fraction of ground truth people dropped")
    ap.add_argument("--extra", default=0.05, type=float, help="false positives per ground truth person")
    ap.add_argument("--tau", default=1, type=float, help="LocSim tau in metres")
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()
    metrics = run(args.gt, args.out, args.sigma, args.miss, args.extra, args.tau, args.seed)
    for typ, m in metrics.items():
        print(typ, {k: round(v, 3) for k, v in m.items()})
