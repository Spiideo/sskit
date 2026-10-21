"""Tests of sskit.pose: the eval_3d matching and helpers, orientation errors, the Neck convention
check, VoxelPose AP_K and the visibility split. numpy/scipy only.

    python -m pytest tests/test_pose.py -v
"""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sskit import pose as pm


def rotz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def skeleton(rng, center=(0, 0, 0)):
    """A random but plausible BODY-25 skeleton (25, 3): hips, shoulders and head laid out so
    that every orientation axis has a well-defined direction."""
    s = rng.normal(0, 0.3, (25, 3))
    s[9], s[12] = [-0.1, 0, 1.0], [0.1, 0, 1.0]           # RHip, LHip
    s[2], s[5] = [-0.2, 0, 1.5], [0.2, 0, 1.5]            # shoulders
    s[17], s[18], s[0] = [-0.07, 0, 1.75], [0.07, 0, 1.75], [0, 0.1, 1.75]   # ears, nose
    return s + np.asarray(center, float)


def voxelpose_eval_list_to_ap(eval_list, total_gt, threshold):
    """VoxelPose's reference, lib/dataset/panoptic.py (Microsoft, MIT licence), verbatim."""
    eval_list.sort(key=lambda k: k["score"], reverse=True)
    total_num = len(eval_list)
    tp = np.zeros(total_num)
    fp = np.zeros(total_num)
    gt_det = []
    for i, item in enumerate(eval_list):
        if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
            tp[i] = 1
            gt_det.append(item["gt_id"])
        else:
            fp[i] = 1
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (total_gt + 1e-5)
    precise = tp / (tp + fp + 1e-5)
    for n in range(total_num - 2, -1, -1):
        precise[n] = max(precise[n], precise[n + 1])
    precise = np.concatenate(([0], precise, [0]))
    recall = np.concatenate(([0], recall, [1]))
    index = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])
    return ap, recall[-2]


class OrientationTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_axes(self):
        s = skeleton(self.rng)
        ax = pm.orientation_axes(s)
        assert_allclose(ax["hips"], s[12] - s[9])
        assert_allclose(ax["shoulders"], s[5] - s[2])
        assert_allclose(ax["head"], s[18] - s[17])
        self.assertEqual(pm.orientation_axes(np.stack([s, s]))["hips"].shape, (2, 3))

    def test_rotation_about_z(self):
        """A skeleton turned by phi about the vertical axis has angle = heading = phi for every
        axis, wherever it stands (translation invariance)."""
        s = skeleton(self.rng)
        for phi in (0, 10, 45, 90, 135, 180, -30, -170):
            t = s @ rotz(phi).T + [12.0, -3.0, 0.0]
            e = pm.orientation_errors(t[None], s[None])
            for name in pm.BODY25_ORIENTATION_AXES:
                assert_allclose(e[name]["angle"], [abs(phi)], atol=1e-9)
                assert_allclose(e[name]["heading"], [abs(phi)], atol=1e-9)

    def test_tilt_has_no_heading(self):
        """Tilting the hip axis out of the ground plane changes the 3D angle but not the heading."""
        s = skeleton(self.rng)
        t = s.copy()
        t[12, 2] += 0.2       # LHip raised: hips vector (0.2, 0, 0.2) vs (0.2, 0, 0): 45 deg
        e = pm.orientation_errors(t[None], s[None])
        assert_allclose(e["hips"]["angle"], [45.0], atol=1e-9)
        assert_allclose(e["hips"]["heading"], [0.0], atol=1e-9)
        assert_allclose(e["shoulders"]["angle"], [0.0], atol=1e-9)

    def test_wrap(self):
        a = np.array([[1.0, 0, 0]])
        for b, exp in (([[-1.0, 1e-9, 0]], 180.0), ([[0, -1.0, 0]], 90.0), ([[1.0, 1.0, 0]], 45.0)):
            assert_allclose(pm.heading_between(a, np.array(b)), [exp], atol=1e-6)
            assert_allclose(pm.angle_between(a, np.array(b)), [exp], atol=1e-6)
        self.assertTrue(np.isfinite(pm.angle_between(a, np.zeros((1, 3)))).all())   # degenerate axis

    def test_eval_orientation(self):
        s = np.stack([skeleton(self.rng) for _ in range(4)])
        t = np.stack([x @ rotz(phi).T for x, phi in zip(s, (10, 20, 30, 40))])
        lines = []
        m = pm.eval_orientation(t, s, lines.append)
        self.assertEqual(m["n"], 4)
        assert_allclose(m["hips_angle_mean"], 25.0, atol=1e-9)
        assert_allclose(m["head_heading_median"], 25.0, atol=1e-9)
        assert_allclose(m["hips_flipped"], 0.0)
        t[0] = t[0] @ rotz(180).T                                   # one of four turned around
        m = pm.eval_orientation(t, s, lines.append)
        self.assertEqual(len(lines), 8)
        assert_allclose(m["shoulders_flipped"], 0.25)
        assert_allclose(m["shoulders_heading_mean_unflipped"], 30.0, atol=1e-9)
        assert_allclose(m["shoulders_heading_mean"], (170 + 20 + 30 + 40) / 4, atol=1e-9)   # 10 + 180 -> 170 wrapped
        self.assertEqual(m["axes"]["head"], (["REar"], ["LEar"]))
        self.assertEqual(pm.eval_orientation(s[:0], s[:0], lambda s: None)["n"], 0)


class APKTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_against_voxelpose_reference(self):
        for trial in range(20):
            n_gt, n = self.rng.integers(1, 12), self.rng.integers(0, 30)
            dist = self.rng.uniform(0, 0.3, n)
            score = self.rng.choice(np.linspace(0, 1, 7), n)         # ties on purpose
            gt_id = self.rng.integers(0, n_gt, n)
            items = [dict(mpjpe=float(d), score=float(s), gt_id=int(g)) for d, s, g in zip(dist, score, gt_id)]
            for thr in (0.05, 0.1, 0.15):
                ap, rec = pm.voxelpose_ap(dist, score, gt_id, n_gt, thr)
                ap_ref, rec_ref = voxelpose_eval_list_to_ap(list(items), n_gt, thr)
                assert_allclose(ap, ap_ref, atol=1e-12)
                assert_allclose(rec, rec_ref, atol=1e-12)

    def test_hand_computed(self):
        # 3 GT; predictions ranked by score: tp (gt 0), fp (gt 0 again), tp (gt 1), fp (dist too large), gt 2 missed
        dist, score, gt_id = [0.02, 0.01, 0.04, 0.5], [0.9, 0.8, 0.7, 0.6], [0, 0, 1, 2]
        ap, rec = pm.voxelpose_ap(dist, score, gt_id, 3, 0.05)
        # precision at the two recall steps (1/3, 2/3): 1 and 2/3 -> AP = (1 + 2/3) / 3
        assert_allclose(ap, (1 + 2 / 3) / 3, atol=1e-4)
        assert_allclose(rec, 2 / 3, atol=1e-4)
        ap, rec = pm.voxelpose_ap(dist, score, gt_id, 3, 1.0)       # loose: gt 2 found too, 2nd is still a duplicate
        # precisions 1, 1/2, 2/3, 3/4 at recall 1/3, 1/3, 2/3, 1 -> interpolated 1, 3/4, 3/4
        assert_allclose(ap, (1 + 3 / 4 + 3 / 4) / 3, atol=1e-4)
        assert_allclose(rec, 1.0, atol=1e-4)
        self.assertEqual(pm.voxelpose_ap([], [], [], 3, 0.05), (0.0, 0.0))
        self.assertEqual(pm.voxelpose_ap([0.01], [1.0], [-1], 3, 0.05)[0], 0.0)   # no GT overlap: fp

    def coco_scene(self):
        """Two images; GT skeletons and detections with controlled errors: a root-aligned error
        on the non-hip joints, optionally a uniform shift (placement error) of the whole body."""
        gt_anns, dets, kp3d = [], [], []
        people = {1: [(0, 0), (3, 0), (6, 0)], 2: [(0, 5), (2, 5)]}
        aid = 0
        for img, pos in people.items():
            for i, (x, y) in enumerate(pos):
                s = skeleton(self.rng, (x, y, 0))
                gt_anns.append(dict(id=aid, image_id=img, bbox=[100 * x, 100 * y, 40, 90],
                                    keypoints_3d=np.c_[s, np.ones(25)].ravel().tolist()))
                aid += 1
        gt = dict(images=[dict(id=1), dict(id=2)], annotations=gt_anns)

        def det(img, ann, score, err, dx=0.0, bbox=None):
            """Prediction with a root-aligned MPJPE of `err` m: the 22 scored non-hip joints are
            shifted along x by err * 24 / 22 (the hips, and with them the root, stay put)."""
            s = pm._kp3d(ann) + [dx, 0, 0]
            s[shifted, 0] += err * 24 / 22
            dets.append(dict(image_id=img, score=score, bbox=bbox or ann["bbox"]))
            kp3d.append(dict(image_id=img, keypoints_3d=s.tolist()))
        shifted = [j for j in pm.joints_used() if j not in pm.ROOT_JOINTS]
        det(1, gt_anns[0], 0.9, 0.03, dx=0.5)       # 3 cm root-aligned, 50 cm misplaced (still nearest GT 0)
        det(1, gt_anns[1], 0.8, 0.12)               # 12 cm
        det(1, gt_anns[1], 0.5, 0.01)               # duplicate of GT 1, better but lower score
        det(2, gt_anns[3], 0.7, 0.07)               # 7 cm
        det(2, gt_anns[4], 0.3, 0.0, dx=0.3)        # exact pose, 30 cm misplaced; GT 2 missed
        return gt, dets, kp3d

    def test_eval_list(self):
        gt, dets, kp3d = self.coco_scene()
        dist, score, gt_id, n_gt = pm.pose_eval_list(gt, dets, kp3d, pm.joints_used())
        self.assertEqual(n_gt, 5)
        self.assertEqual(gt_id.tolist(), [0, 1, 1, 3, 4])            # nearest by global MPJPE
        assert_allclose(dist["global"], [0.53, 0.12, 0.01, 0.07, 0.3], atol=1e-9)
        assert_allclose(dist["root"], [0.03, 0.12, 0.01, 0.07, 0.0], atol=1e-9)
        assert_allclose(score, [0.9, 0.8, 0.5, 0.7, 0.3])
        # a detection far from everyone is assigned the nearest GT anyway (VoxelPose), and is then a fp
        far = dict(kp3d[0], keypoints_3d=(np.array(kp3d[0]["keypoints_3d"]) + [50, 0, 0]).tolist())
        d2, _, g2, _ = pm.pose_eval_list(gt, dets[:1], [far], pm.joints_used())
        self.assertEqual(g2.tolist(), [2])                           # GT 2 at x = 6 is the nearest
        self.assertGreater(d2["global"][0], 44)
        self.assertGreater(d2["root"][0], 0.1)                       # another person's pose

    def test_eval_ap_k(self):
        gt, dets, kp3d = self.coco_scene()
        lines = []
        m = pm.eval_ap_k(gt, dets, kp3d, lines.append)
        self.assertEqual((m["n_gt"], m["n_pred"]), (5, 5))
        g, r = m["global"], m["root"]
        # ranked by score: 0.9 (gt0: 53 / 3 cm), 0.8 (gt1: 12), 0.7 (gt3: 7), 0.5 (gt1: 1), 0.3 (gt4: 30 / 0); gt2 missed
        # global, K = 5: fp fp fp tp fp -> precision 1/4 at recall 1/5
        assert_allclose(g["AP5"], 0.25 / 5, atol=1e-4)
        assert_allclose(g["recall5"], 1 / 5, atol=1e-4)
        # global, K = 10: fp fp tp tp fp -> precisions 1/3, 1/2 -> interpolated 1/2, 1/2
        assert_allclose(g["AP10"], (0.5 + 0.5) / 5, atol=1e-4)
        # global, K = 15: fp tp tp fp(gt1 claimed) fp -> precisions 1/2, 2/3 -> interpolated 2/3, 2/3
        assert_allclose(g["AP15"], (2 / 3 + 2 / 3) / 5, atol=1e-4)
        assert_allclose(g["recall15"], 2 / 5, atol=1e-4)
        # root, K = 5: tp fp fp tp tp -> precisions 1, 1/2, 3/5 -> interpolated 1, 3/5, 3/5
        assert_allclose(r["AP5"], (1 + 0.6 + 0.6) / 5, atol=1e-4)
        assert_allclose(r["recall5"], 3 / 5, atol=1e-4)
        # root, K = 10: tp fp tp tp tp -> precisions 1, 2/3, 3/4, 4/5 -> interpolated 1, 4/5, 4/5, 4/5
        assert_allclose(r["AP10"], (1 + 3 * 0.8) / 5, atol=1e-4)
        # root, K = 15: tp tp tp fp tp -> precisions 1, 1, 1, 4/5
        assert_allclose(r["AP15"], (3 + 0.8) / 5, atol=1e-4)
        assert_allclose(r["recall15"], 4 / 5, atol=1e-4)
        for x in (g, r):
            assert_allclose(x["mAP"], (x["AP5"] + x["AP10"] + x["AP15"]) / 3)
        self.assertEqual(len(lines), 3)
        self.assertEqual(len(m["joints_averaged"]), 24)

    def test_eval_3d(self):
        gt, dets, kp3d = self.coco_scene()
        cam = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, -30.0]]
        gt = dict(gt, images=[dict(im, camera_matrix=cam) for im in gt["images"]])
        lines = []
        m = pm.eval_3d(gt, dets, kp3d, 0.5, lines.append)
        self.assertEqual((m["n_gt"], m["n_pred"], m["n_matched"]), (5, 5, 4))
        P, G, _ = pm.matched_pairs(gt, dets, kp3d, 0.5)
        use = pm.joints_used()
        for key, align in (("mpjpe_global", "none"), ("mpjpe_local", "root"), ("pa_mpjpe", "procrustes")):
            assert_allclose(m[key], pm.pose_distance(P, G, use, align).mean(), atol=1e-12)
        assert_allclose(m["mpjpe_local_median"], np.median(pm.pose_distance(P, G, use, "root")), atol=1e-12)
        assert_allclose(m["per_joint_local"]["RHip"], 0.0, atol=1e-12)                  # hips never move
        root = np.linalg.norm(P[:, pm.ROOT_JOINTS].mean(1) - G[:, pm.ROOT_JOINTS].mean(1), axis=1)
        assert_allclose(m["root_error_mean"], root.mean(), atol=1e-12)
        self.assertEqual(len(m["joints_averaged"]), 24)
        self.assertEqual(len(lines), 5)
        self.assertEqual(pm.eval_3d(gt, [], [], 0.5, lambda s: None)["n_matched"], 0)

    def test_matched_pairs(self):
        gt, dets, kp3d = self.coco_scene()
        P, G, (n_gt, n_pred) = pm.matched_pairs(gt, dets, kp3d, 0.5)
        self.assertEqual((n_gt, n_pred, P.shape, G.shape), (5, 5, (4, 25, 3), (4, 25, 3)))
        use = pm.joints_used()
        d = np.sort(pm.pose_distance(P, G, use, "root"))
        assert_allclose(d[:3], [0.0, 0.03, 0.07], atol=1e-9)
        self.assertTrue(np.isclose(d[3], 0.01) or np.isclose(d[3], 0.12))   # either box of GT 1 (IoU tie)
        pa = pm.pose_distance(P, G, use, "procrustes")
        self.assertTrue(np.all(pa <= pm.pose_distance(P, G, use, "root") + 1e-9))   # alignment can only help
        assert_allclose(pm.pose_distance(G, G, use, "procrustes"), 0, atol=1e-9)


class NeckTest(unittest.TestCase):
    """Neck offsets in the body frame and the relocation to another convention."""

    def test_offsets_and_relocation(self):
        g = skeleton(np.random.default_rng(3))             # shoulders at z = 1.5, x = -+0.2: up = +z, left = +x, back = +y
        g[1] = [0.0, 0.0, 1.56]                              # SMPL-X style: 6 cm straight above the shoulder midpoint
        p = g.copy()
        p[1] = [0.0, -0.03, 1.54]                            # MHR style: 4 cm up, 3 cm forward (-y is forward)
        assert_allclose(pm.neck_offsets(g[None]), [[0.06, 0.0, 0.0]], atol=1e-12)
        assert_allclose(pm.neck_offsets(p[None]), [[0.04, 0.0, -0.03]], atol=1e-12)
        moved = (rotz(37) @ p.T).T + [3, -2, 0.5]
        assert_allclose(pm.neck_offsets(moved[None]), [[0.04, 0.0, -0.03]], atol=1e-12)   # rotation/translation invariant
        assert_allclose(pm.relocate_neck(p[None], [0.06, 0, 0])[0, 1], g[1], atol=1e-12)
        m = pm.eval_neck(np.stack([p, moved]), np.stack([g, g]), log=lambda s: None)
        assert_allclose(m["pred_offset"], [0.04, 0.0, -0.03], atol=1e-12)
        assert_allclose(m["gt_offset"], [0.06, 0.0, 0.0], atol=1e-12)
        assert_allclose(m["pred_offset_norm"], 0.05, atol=1e-12)
        self.assertEqual(m["n"], 2)
        # the first pair: same skeleton, Neck 2 cm lower and 3 cm forward -> error sqrt(0.02^2 + 0.03^2), zero after relocation
        m1 = pm.eval_neck(p[None], g[None], log=lambda s: None)
        assert_allclose(m1["neck_local"], np.hypot(0.02, 0.03), atol=1e-12)
        assert_allclose(m1["neck_local_relocated"], 0.0, atol=1e-12)
        assert_allclose(m1["shoulder_mid_local"], 0.0, atol=1e-12)


class VisibilityTest(unittest.TestCase):
    """Per-joint errors split by the GT visibility flag, GT subsets for the AP metrics and the
    visible-joint AP_K."""

    def scene(self):
        rng = np.random.default_rng(5)
        gt_anns, dets, kp3d = [], [], []
        flags = np.full((3, 25), 2)
        flags[0, [4, 7, 19]] = 1                     # person 0: wrists and LBigToe occluded
        flags[1, [0, 15, 16, 17, 18]] = 0            # person 1: face outside the image
        flags[2, 1:] = 1                             # person 2: everything but the Nose occluded (hips too)
        for i in range(3):
            s = skeleton(rng, (3 * i, 0, 0))
            uv = rng.uniform(100, 500, (25, 2))
            gt_anns.append(dict(id=i, image_id=1, bbox=[300 * i, 0, 40, 90], area=3600.0,
                                keypoints=np.c_[uv, flags[i]].ravel().tolist(),
                                keypoints_3d=np.c_[s, np.ones(25)].ravel().tolist()))
            p = s.copy()
            err = np.where(flags[i] == 2, 0.01, np.where(flags[i] == 1, 0.05, 0.20))   # per-joint 3D error by class
            p[:, 0] += err
            dets.append(dict(image_id=1, score=0.9 - 0.1 * i, bbox=gt_anns[-1]["bbox"],
                             keypoints=np.c_[uv + np.c_[10 * (flags[i] != 2), np.zeros(25)], np.ones(25)].ravel().tolist()))
            kp3d.append(dict(image_id=1, keypoints_3d=p.tolist()))
        cam = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, -30.0]]            # eval_3d needs the camera position
        return dict(images=[dict(id=1, camera_matrix=cam)], annotations=gt_anns), dets, kp3d, flags

    def test_split(self):
        gt, dets, kp3d, flags = self.scene()
        m = pm.eval_visibility(gt, dets, kp3d, lambda s: None)
        use = pm.joints_used()
        self.assertEqual((m["n"], m["n_gt"], m["n_pred"]), (3, 3, 3))
        f = flags[:, use]
        self.assertEqual(m["all"]["n_joints"], 3 * 24)
        self.assertEqual(m["visible"]["n_joints"], int((f == 2).sum()))
        self.assertEqual(m["occluded"]["n_joints"], int((f == 1).sum()))
        self.assertEqual(m["outside"]["n_joints"], int((f == 0).sum()))
        assert_allclose(m["visible"]["fraction"] + m["occluded"]["fraction"] + m["outside"]["fraction"], 1.0)
        # global errors are the per-class shifts along x; the root-aligned error of person 2 (hips shifted 5 cm) differs
        assert_allclose(m["visible"]["mpjpe_global"], 0.01, atol=1e-12)
        assert_allclose(m["occluded"]["mpjpe_global"], 0.05, atol=1e-12)
        assert_allclose(m["outside"]["mpjpe_global"], 0.20, atol=1e-12)
        assert_allclose(m["all"]["mpjpe_global"], np.where(f == 2, 0.01, np.where(f == 1, 0.05, 0.2)).mean(), atol=1e-12)
        assert_allclose(m["visible"]["error_2d"], 0.0, atol=1e-12)
        assert_allclose(m["occluded"]["error_2d"], 10.0, atol=1e-9)
        assert_allclose(m["visible"]["oks_term"], 1.0, atol=1e-12)
        # OKS term of a 10 px error at scale s^2 = 3600 and k = 2 sigma: exp(-100 / (2 * 3600 * 4 sigma^2))
        sig = np.array([pm.BODY25_SIGMAS[j] for j in use for i in range(3) if f[i, use.index(j)] == 1])
        assert_allclose(m["occluded"]["oks_term"], np.exp(-100 / (2 * 3600 * (2 * sig) ** 2)).mean(), atol=1e-9)
        self.assertTrue(np.isnan(m["outside"]["per_joint_local"]["RWrist"]))   # no RWrist outside the image
        assert_allclose(m["outside"]["per_joint_fraction"]["Nose"], 1 / 3)
        assert_allclose(m["occluded"]["per_joint_global"]["RWrist"], 0.05, atol=1e-12)
        assert_allclose(m["people"]["any_occluded"], 2 / 3)
        assert_allclose(m["people"]["any_outside"], 1 / 3)
        assert_allclose(m["people"]["occluded_fraction"], (3 / 24 + 0 + 23 / 24) / 3)      # MidHip is not scored
        o = m["orientation"]
        self.assertEqual((o["hips"]["visible"]["n"], o["hips"]["hidden"]["n"]), (2, 1))
        self.assertEqual((o["head"]["visible"]["n"], o["head"]["hidden"]["n"]), (1, 2))
        assert_allclose(o["hips"]["visible"]["angle_mean"], 0.0, atol=1e-9)       # both hips shifted alike
        # the "all" entry reproduces eval_3d
        e = pm.eval_3d(gt, dets, kp3d, 0.5, lambda s: None)
        for k in ("mpjpe_global", "mpjpe_local", "pa_mpjpe"):
            assert_allclose(m["all"][k], e[k], atol=1e-12)
        self.assertEqual(pm.eval_visibility(dict(gt, annotations=[]), [], [], lambda s: None)["n"], 0)

    def test_subset(self):
        gt, dets, kp3d, flags = self.scene()
        sub = pm.visibility_subset(gt, {2})
        v = np.array([a["keypoints"][2::3] for a in sub["annotations"]])
        self.assertTrue(((v == 2) == (flags == 2)).all())
        self.assertEqual([a["num_keypoints"] for a in sub["annotations"]], [22, 20, 1])
        self.assertEqual(gt["annotations"][2]["keypoints"][5], 1)                     # the input is untouched
        occ = pm.visibility_subset(gt, {1})
        self.assertEqual([a["num_keypoints"] for a in occ["annotations"]], [3, 0, 24])
        self.assertEqual(np.array(occ["annotations"][0]["keypoints"][2::3]).max(), 1)

    def test_ap_k_flags(self):
        gt, dets, kp3d, flags = self.scene()
        use = pm.joints_used()
        d_all, _, g_all, n_all = pm.pose_eval_list(gt, dets, kp3d, use)
        d_vis, _, g_vis, n_vis = pm.pose_eval_list(gt, dets, kp3d, use, flags={2})
        self.assertEqual((n_all, n_vis), (3, 3))
        self.assertEqual(g_vis.tolist(), [0, 1, 2])
        assert_allclose(d_vis["global"], [0.01, 0.01, 0.01], atol=1e-12)              # only the 1 cm joints are scored
        assert_allclose(d_all["global"][2], 0.05 * 23 / 24 + 0.01 / 24, atol=1e-12)
        d_occ, _, g_occ, n_occ = pm.pose_eval_list(gt, dets, kp3d, use, flags={1})
        self.assertEqual(n_occ, 2)                                                    # person 1 has no occluded joint
        self.assertEqual(g_occ.tolist()[0], 0)
        m = pm.eval_ap_k(gt, dets, kp3d, lambda s: None, flags={2})
        self.assertEqual(m["flags"], [2])
        assert_allclose(m["global"]["AP5"], 1.0, atol=1e-4)
        m_all = pm.eval_ap_k(gt, dets, kp3d, lambda s: None)
        self.assertNotIn("flags", m_all)
        self.assertLess(m_all["global"]["AP5"], 1.0)                                  # person 2's occluded joints fail K = 5 cm
        # no flags: the masked path equals the plain MPJPE
        P, G, _ = pm.matched_pairs(gt, dets, kp3d)
        mask = pm.joint_mask(gt["annotations"], use)
        assert_allclose(pm.masked_distance(P, G, mask), pm.pose_distance(P, G, use, "none"), atol=1e-12)
        assert_allclose(pm.masked_distance(P, G, mask, "root"), pm.pose_distance(P, G, use, "root"), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
