"""BODY-25 3D pose evaluation: constants of the OpenPose BODY-25 skeleton, the MPJPE evaluation
of IoU-matched people (`eval_3d`), and on top of it the orientation error of the body axes
(`eval_orientation`), VoxelPose's AP_K (`eval_ap_k`), the Neck convention check (`eval_neck`)
and the split of the errors by the visibility flag of the GT joints (`eval_visibility`).
numpy/scipy only (the camera position of `eval_3d` comes from `sskit.camera`); every metric
takes a `log` callback and returns a metrics dict.

Ground truth is a COCO keypoint file with, per annotation, `keypoints` (25 x [u, v, flag],
pixels) and `keypoints_3d` (25 x [x, y, z, 1], world metres), as written by
`convert_pose.py --coco-out`; the images carry a `camera_matrix`. Detections are COCO results
(`image_id`, `score`, `bbox`, `keypoints`) with the 3D skeletons as a parallel list of dicts
holding `keypoints_3d` (25 x [x, y, z]), the json files that SAM3DBody/synloc.py writes.

Orientation error
    The angle, in degrees, between the predicted and the ground-truth direction of a body
    axis: hips (RHip -> LHip), shoulders (RShoulder -> LShoulder) and head (REar -> LEar),
    all three the left-right axis of the body part. `angle` is the full 3D angle between the two
    vectors, `heading` the difference of their azimuths in the ground plane (z up), the
    quantity that matters for "which way is the player facing". Both are invariant to the
    placement, so they are a root-relative pose metric like the local MPJPE, and they are
    computed over the same IoU-matched pairs.

VoxelPose AP_K (Tu, Wang and Zeng, ECCV 2020), global and root-aligned
    VoxelPose scores a multi-person 3D pose detector like a detector: every predicted pose
    is assigned the ground-truth pose of its image with the smallest MPJPE in world
    coordinates, the predictions of all images are ranked by score, a prediction is a true
    positive if that MPJPE is below K and its ground-truth pose has not been claimed by a
    higher-scoring prediction, and AP_K is the area under the interpolated precision-recall
    curve (VOC style). Misses cost recall and spurious skeletons cost precision, which the
    matched-only MPJPEs never do. `eval_ap_k` reports this exactly (`global`: the metric of
    the VoxelPose paper, K = 25 .. 150 mm there) and, for the same assignment, with the
    *root-aligned* MPJPE as the test (`root`: hip-mean root, same joints as the local
    MPJPE), which scores the pose and the detection but forgives the placement, as long as
    the placement is good enough for the assignment. `voxelpose_ap` is a faithful port of
    VoxelPose's `_eval_list_to_ap`, including its greedy claim rule: a detection whose
    nearest ground-truth pose was claimed by a higher-scoring detection is a false
    positive even if another unclaimed pose is within K (several detections may pick the
    same pose, no detection picks several). The one deviation is that detections in images
    without any ground-truth pose count as false positives, whereas VoxelPose skips such
    images.

Neck convention
    Neither the SMPL-X based ground truth nor most predictors put the Neck at the OpenPose
    shoulder midpoint. `eval_neck` records where the Neck of each side sits relative to the
    shoulder midpoint in a body frame (up, left, back) and the Neck error with the predicted
    Neck relocated to the ground-truth offset, i.e. what the predictor's own convention costs.

Visibility split
    The SynLoc keypoint files flag every joint as visible (v = 2), occluded by a body according
    to a z-buffer of the fitted SMPL-X meshes (v = 1) or outside the image (v = 0).
    `eval_visibility` scores the IoU-matched pairs of eval_3d per joint and splits the joints by
    that flag: global, root-aligned and Procrustes-aligned 3D error (the alignments use all the
    scored joints, as in eval_3d, only the averages are split), the 2D error in pixels and the
    per-joint OKS term, plus the orientation errors of the people whose axis joints are all
    visible against the others. `visibility_subset` restricts a COCO ground truth to the joints of
    one class, so that sskit.coco.coco_eval scores the keypoint AP and the 3D-LocSim AP over
    those joints only (the others are unlabelled, as coco_eval's `exclude` does), and
    `eval_ap_k(..., flags={2})` tests and assigns on the visible joints only.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from sskit.camera import camera_position


BODY25_NAMES = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
                "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
# COCO OKS sigmas for the 17 COCO joints, COCO-WholeBody foot sigmas for toes/heels,
# shoulder-like for the neck and hip-like for MidHip.
BODY25_SIGMAS = np.array([.026, .079, .079, .072, .062, .079, .072, .062, .107, .107, .087, .089,
                          .107, .087, .089, .025, .025, .035, .035, .068, .066, .066, .068, .066, .066])
BODY25_PAIRS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10),
                (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16),
                (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (11, 23), (11, 24)]

# From FIFA starter kit, https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026:
# The 15 FIFA15 joints as BODY25 indices.
BODY25_TO_FIFA15 = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]
# The 17 COCO joints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles; left before right)
# as BODY25 indices.
BODY25_TO_COCO17 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]

ROOT_JOINTS = [9, 12]  # RHip, LHip: the root is their mean on both sides
NECK, SHOULDERS = 1, (2, 5)    # Neck; RShoulder, LShoulder

# Orientation axes, name -> (from joints, to joints): the axis is mean(to) - mean(from), BODY-25 indices.
# All three are the right-to-left axis of the body part, so the errors are comparable and a 180 degree
# error means the part is turned around. A facing axis, e.g. "face": ((17, 18), (0,)), is a one-line addition.
BODY25_ORIENTATION_AXES = {
    "hips": ((9,), (12,)),          # RHip -> LHip
    "shoulders": ((2,), (5,)),      # RShoulder -> LShoulder
    "head": ((17,), (18,)),         # REar -> LEar
}

VISIBILITY_CLASSES = {"visible": 2, "occluded": 1, "outside": 0}   # COCO flag of each class in the SynLoc keypoint files


# --- shared helpers ---------------------------------------------------------------------------
def joints_used(midhip="hipmean"):
    """Joints entering the averages: all 25, or without MidHip (8) for hip-midpoint predictors."""
    return [j for j in range(25) if midhip == "pelvis" or j != 8]


def _kp3d(a, key="keypoints_3d"):
    """(25, 3) float array of a GT annotation ((25, 4) rows) or a detection ((25, 3) rows)."""
    kp = np.asarray(a[key], float)
    kp = kp.reshape(-1, 4 if kp.size % 3 else 3) if kp.ndim == 1 else kp
    return kp[:, :3]


def _kp2d(a):
    """(25, 3) float array [u, v, flag] of the COCO `keypoints` of a GT annotation or a detection."""
    return np.asarray(a["keypoints"], float).reshape(-1, 3)


def by_image(gt, dets, kp3d):
    """GT annotations and (detection, 3D skeleton) pairs grouped by image id, in input order."""
    gt_by_img, det_by_img = {}, {}
    for a in gt["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)
    for d, k in zip(dets, kp3d):
        det_by_img.setdefault(d["image_id"], []).append((d, k))
    return gt_by_img, det_by_img


def matched_triples(gt, dets, kp3d, iou_thr=0.5):
    """The matching of eval_3d: Hungarian assignment on box IoU per image, pairs with
    IoU >= iou_thr kept. Returns the list of (GT annotation, detection, 3D skeleton dict)
    triples of the matched pairs, in image order, plus (n_gt, n_pred)."""
    gt_by_img, det_by_img = by_image(gt, dets, kp3d)
    out = []
    for img_id, gts in gt_by_img.items():
        ds = det_by_img.get(img_id, [])
        if not ds:
            continue
        iou = box_iou(np.array([a["bbox"] for a in gts], float), np.array([d["bbox"] for d, _ in ds], float))
        rows, cols = linear_sum_assignment(-iou)
        for r, c in zip(rows, cols):
            if iou[r, c] >= iou_thr:
                out.append((gts[r], ds[c][0], ds[c][1]))
    return out, (len(gt["annotations"]), len(dets))


def _skeletons(triples):
    """Predicted and GT skeletons of matched triples as (N, 25, 3) arrays."""
    P = np.array([_kp3d(k) for _, _, k in triples]).reshape(-1, 25, 3)
    G = np.array([_kp3d(a) for a, _, _ in triples]).reshape(-1, 25, 3)
    return P, G


def matched_pairs(gt, dets, kp3d, iou_thr=0.5):
    """The predicted and ground-truth skeletons of the pairs of `matched_triples` (the pairs
    eval_3d scores) as (N, 25, 3) arrays plus (n_gt, n_pred)."""
    triples, n = matched_triples(gt, dets, kp3d, iou_thr)
    P, G = _skeletons(triples)
    return P, G, n


def joint_errors(pred, gt, use, align="none"):
    """Per-joint distance between skeletons pred (..., 25, 3) and gt (..., 25, 3), (..., 25):
    align="none" in the given coordinates, "root" after subtracting the hip-mean root of each
    skeleton, "procrustes" after a similarity alignment of pred onto gt fitted on the joints
    `use` (one pair at a time)."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    if align == "root":
        pred = pred - pred[..., ROOT_JOINTS, :].mean(-2, keepdims=True)
        gt = gt - gt[..., ROOT_JOINTS, :].mean(-2, keepdims=True)
    elif align == "procrustes":
        flat = pred.reshape(-1, 25, 3)
        pred = np.array([procrustes(p, g, use) for p, g in zip(flat, gt.reshape(-1, 25, 3))]).reshape(pred.shape)
    elif align != "none":
        raise ValueError(f"align={align!r}, expected none, root or procrustes")
    return np.linalg.norm(pred - gt, axis=-1)


def pose_distance(pred, gt, use, align="root"):
    """MPJPE between skeletons pred (..., 25, 3) and gt (..., 25, 3) over the joints `use`:
    align="none" in the given coordinates, "root" after subtracting the hip-mean root of
    each skeleton (the local MPJPE of eval_3d), "procrustes" after a similarity alignment of
    pred onto gt (PA-MPJPE, one pair at a time)."""
    return joint_errors(pred, gt, use, align)[..., use].mean(-1)


# --- MPJPE ------------------------------------------------------------------------------------
def eval_3d(gt, dets, kp3d, iou_thr, log, midhip="hipmean"):
    """3D errors of the people matched by box IoU (Hungarian assignment per image, pairs with
    IoU >= iou_thr; `matched_pairs` gives the same pairs): global MPJPE (world coordinates),
    local MPJPE (root-aligned, root = hip mean), PA-MPJPE, the root position and the
    distance-to-camera errors, the FIFA-15 subset and the per-joint errors, in metres. The
    predicted MidHip is left out of the averages unless midhip="pelvis" (predictors that
    output a real pelvis)."""
    use = joints_used(midhip)   # joints entering the averages
    triples, (n_gt, n_pred) = matched_triples(gt, dets, kp3d, iou_thr)
    n_matched = len(triples)
    m = dict(n_gt=n_gt, n_pred=n_pred, n_matched=n_matched, match_iou_thr=iou_thr, root="mean of RHip and LHip",
             midhip=midhip, joints_averaged=[BODY25_NAMES[j] for j in use])
    log(f"== 3D (BODY25, metres): {n_matched} of {n_gt} GT people matched to {n_pred} predictions at IoU >= {iou_thr}; "
        f"root = hip mean, MidHip {'excluded from the averages' if midhip == 'hipmean' else 'scored against the GT pelvis'}")
    if n_matched == 0:
        return m
    P, G = _skeletons(triples)
    cam_pos = {im["id"]: camera_position(im["camera_matrix"]).numpy()[:, 0] for im in gt["images"]}
    err_g, err_l, err_pa = joint_errors(P, G, use, "none"), joint_errors(P, G, use, "root"), joint_errors(P, G, use, "procrustes")
    p_root, g_root = P[:, ROOT_JOINTS].mean(1), G[:, ROOT_JOINTS].mean(1)
    root_err = p_root - g_root
    gt_depth = np.array([np.linalg.norm(g - cam_pos[a["image_id"]]) for g, (a, _, _) in zip(g_root, triples)])
    depth_err = np.array([np.linalg.norm(p - cam_pos[a["image_id"]]) for p, (a, _, _) in zip(p_root, triples)]) - gt_depth
    m.update(mpjpe_global=float(err_g[:, use].mean()), mpjpe_local=float(err_l[:, use].mean()), pa_mpjpe=float(err_pa[:, use].mean()),
             mpjpe_global_median=float(np.median(err_g[:, use].mean(1))), mpjpe_local_median=float(np.median(err_l[:, use].mean(1))),
             root_error_mean=float(np.linalg.norm(root_err, axis=1).mean()),
             root_error_xy_mean=float(np.linalg.norm(root_err[:, :2], axis=1).mean()),
             root_z_bias=float(root_err[:, 2].mean()),
             depth_error_mean_signed=float(depth_err.mean()), depth_error_mean_abs=float(np.abs(depth_err).mean()),
             gt_distance_mean=float(gt_depth.mean()),
             midhip_vs_gt_pelvis_global=float(err_g[:, 8].mean()), midhip_vs_gt_pelvis_local=float(err_l[:, 8].mean()),
             # FIFA Skeletal Tracking Light: 15 joints, root = hip mean, score = global + 5 * local
             fifa15_mpjpe_global=float(err_g[:, BODY25_TO_FIFA15].mean()), fifa15_mpjpe_local=float(err_l[:, BODY25_TO_FIFA15].mean()),
             fifa15_score=float(err_g[:, BODY25_TO_FIFA15].mean() + 5 * err_l[:, BODY25_TO_FIFA15].mean()),
             per_joint_global=dict(zip(BODY25_NAMES, err_g.mean(0).round(4).tolist())),
             per_joint_local=dict(zip(BODY25_NAMES, err_l.mean(0).round(4).tolist())))
    log(f"  global MPJPE (world coords)      {100 * m['mpjpe_global']:7.1f} cm   (median over people {100 * m['mpjpe_global_median']:.1f} cm)\n"
        f"  local  MPJPE (root-aligned)      {100 * m['mpjpe_local']:7.1f} cm   (median over people {100 * m['mpjpe_local_median']:.1f} cm)\n"
        f"  PA-MPJPE (similarity-aligned)    {100 * m['pa_mpjpe']:7.1f} cm\n"
        f"  root (hip mean) position error   {100 * m['root_error_mean']:7.1f} cm   (ground plane xy {100 * m['root_error_xy_mean']:.1f} cm, "
        f"z bias {100 * m['root_z_bias']:+.1f} cm)\n"
        f"  distance to camera error         {100 * m['depth_error_mean_signed']:+7.1f} cm signed, {100 * m['depth_error_mean_abs']:.1f} cm abs "
        f"(GT people are {m['gt_distance_mean']:.1f} m away on average)\n"
        f"  predicted MidHip vs GT pelvis    {100 * m['midhip_vs_gt_pelvis_local']:7.1f} cm root-aligned"
        + ("   (not in the averages above)" if midhip == "hipmean" else "") + "\n"
        f"  FIFA-15 subset: global MPJPE     {100 * m['fifa15_mpjpe_global']:7.1f} cm, local {100 * m['fifa15_mpjpe_local']:.1f} cm, "
        f"challenge score global + 5 local = {m['fifa15_score']:.3f} m")
    fmt = lambda e: " ".join(f"{100 * v:6.1f}" if j in use else "     -" for j, v in enumerate(e))   # noqa: E731
    log("  per joint (cm)   " + " ".join(f"{n[:6]:>6}" for n in BODY25_NAMES))
    log("    global         " + fmt(err_g.mean(0)))
    log("    local          " + fmt(err_l.mean(0)))
    return m


def box_iou(a, b):
    """a (N, 4), b (M, 4) xywh -> (N, M)."""
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1) - np.maximum(ax0[:, None], bx0), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1) - np.maximum(ay0[:, None], by0), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter + 1e-9)


def procrustes(pred, gt, use):
    """Similarity transform of pred onto gt (both (J, 3)) fitted on the joints in `use`,
    returns all of pred aligned."""
    mp, mg = pred[use].mean(0), gt[use].mean(0)
    p, g = pred[use] - mp, gt[use] - mg
    U, S, Vt = np.linalg.svd(p.T @ g)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (p ** 2).sum()
    return s * (pred - mp) @ R + mg


# --- orientation ------------------------------------------------------------------------------
def orientation_axes(kp, axes=BODY25_ORIENTATION_AXES):
    """Axis vectors of skeletons kp (..., 25, 3): dict name -> (..., 3), mean(to) - mean(from)."""
    kp = np.asarray(kp, float)
    return {name: kp[..., list(to), :].mean(-2) - kp[..., list(frm), :].mean(-2) for name, (frm, to) in axes.items()}


def angle_between(a, b):
    """Angle in degrees between the vectors a and b (..., 3), 0 when parallel, 180 when opposite."""
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def heading_between(a, b):
    """Absolute difference in degrees of the azimuths of a and b (..., 3) in the ground plane
    (x, y; z up), in [0, 180]."""
    d = np.degrees(np.arctan2(a[..., 1], a[..., 0]) - np.arctan2(b[..., 1], b[..., 0]))
    return np.abs((d + 180.0) % 360.0 - 180.0)


def orientation_errors(pred, gt, axes=BODY25_ORIENTATION_AXES):
    """Per-skeleton orientation errors of pred against gt (both (N, 25, 3), already matched):
    dict name -> dict(angle=(N,), heading=(N,)) in degrees. Translation invariant, so the
    placement does not matter."""
    ap, ag = orientation_axes(pred, axes), orientation_axes(gt, axes)
    return {name: dict(angle=angle_between(ap[name], ag[name]), heading=heading_between(ap[name], ag[name])) for name in axes}


def eval_orientation(pred, gt, log=print, axes=BODY25_ORIENTATION_AXES):
    """Mean and median orientation errors over matched pairs (pred, gt: (N, 25, 3), e.g. from
    `matched_pairs`), logged in the style of eval_3d. Returns dict(n=N, axes={name: (from, to)
    joint names}, <name>_angle_mean, <name>_angle_median, <name>_heading_mean,
    <name>_heading_median, <name>_flipped: the fraction of people whose axis heading is more
    than 90 degrees off, i.e. turned the wrong way, and <name>_heading_mean_unflipped over the
    others)."""
    n = len(pred)
    m = dict(n=n, axes={name: ([BODY25_NAMES[j] for j in frm], [BODY25_NAMES[j] for j in to]) for name, (frm, to) in axes.items()})
    log(f"== orientation (degrees, over the {n} matched people): angle between the predicted and GT axis in 3D, "
        f"and its heading difference in the ground plane")
    if n == 0:
        return m
    errs = orientation_errors(pred, gt, axes)
    for name, e in errs.items():
        flipped = e["heading"] > 90
        m.update({f"{name}_angle_mean": float(e["angle"].mean()), f"{name}_angle_median": float(np.median(e["angle"])),
                  f"{name}_heading_mean": float(e["heading"].mean()), f"{name}_heading_median": float(np.median(e["heading"])),
                  f"{name}_flipped": float(flipped.mean()),
                  f"{name}_heading_mean_unflipped": float(e["heading"][~flipped].mean()) if (~flipped).any() else float("nan")})
        frm, to = m["axes"][name]
        log(f"  {name:<10} ({'+'.join(frm)} -> {'+'.join(to)}): angle {m[name + '_angle_mean']:5.1f} deg (median {m[name + '_angle_median']:.1f}), "
            f"heading {m[name + '_heading_mean']:5.1f} deg (median {m[name + '_heading_median']:.1f}); "
            f"{100 * m[name + '_flipped']:.1f} % turned by more than 90 deg, heading {m[name + '_heading_mean_unflipped']:.1f} deg without them")
    return m


# --- Neck convention --------------------------------------------------------------------------
def body_frame(kp):
    """Per-skeleton body frame of BODY-25 skeletons kp (N, 25, 3): axes (N, 3, 3) with rows up
    (hip midpoint -> shoulder midpoint), left (RShoulder -> LShoulder, orthogonalised) and back
    (up x left, i.e. backward in right-handed coordinates), plus the shoulder midpoint (N, 3)."""
    kp = np.asarray(kp, float)
    shm = kp[:, list(SHOULDERS)].mean(1)
    up = shm - kp[:, list(ROOT_JOINTS)].mean(1)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    left = kp[:, SHOULDERS[1]] - kp[:, SHOULDERS[0]]
    left -= (left * up).sum(1, keepdims=True) * up
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    return np.stack([up, left, np.cross(up, left)], 1), shm


def neck_offsets(kp):
    """Offset of the Neck from the shoulder midpoint in the body frame, (N, 3) as (up, left, back)
    in the units of kp. Where a convention puts the Neck: zero for the OpenPose shoulder midpoint,
    straight up for the SMPL-X neck joint."""
    axes, shm = body_frame(kp)
    return np.einsum("nij,nj->ni", axes, np.asarray(kp, float)[:, NECK] - shm)


def relocate_neck(kp, offset):
    """Copy of kp (N, 25, 3) with the Neck put at the shoulder midpoint plus `offset` (up, left,
    back) in each skeleton's body frame, i.e. converted to the convention `offset` describes."""
    axes, shm = body_frame(kp)
    out = np.array(kp, float)
    out[:, NECK] = shm + np.einsum("nij,ni->nj", axes, np.broadcast_to(np.asarray(offset, float), (len(out), 3)))
    return out


def eval_neck(pred, gt, log=print):
    """Neck convention check over matched pairs (pred, gt: (N, 25, 3)): where the Neck of each
    side sits relative to the shoulder midpoint (<side>_offset: mean (up, left, back), metres,
    <side>_offset_std and <side>_offset_norm: the mean distance), the root-aligned Neck error
    (neck_local, as per_joint_local["Neck"] of eval_3d), the error of the shoulder midpoint
    (shoulder_mid_local: the Neck of both sides under the OpenPose convention) and the Neck error
    with the predicted Neck relocated to the mean GT offset (neck_local_relocated): the Neck error
    the predictor would have with the ground-truth convention, so the difference is what its
    own convention costs. Placement independent."""
    n = len(pred)
    m = dict(n=n)
    log(f"== Neck convention (over the {n} matched people): Neck relative to the shoulder midpoint, in each "
        f"skeleton's body frame (up, left, back)")
    if n == 0:
        return m
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    off = {"pred": neck_offsets(pred), "gt": neck_offsets(gt)}
    for side, o in off.items():
        m.update({f"{side}_offset": o.mean(0).tolist(), f"{side}_offset_std": o.std(0).tolist(),
                  f"{side}_offset_norm": float(np.linalg.norm(o, axis=1).mean())})
        log(f"  {side:<4} Neck - shoulder midpoint: ({', '.join(f'{100 * v:+.1f}' for v in m[side + '_offset'])}) cm, "
            f"std ({', '.join(f'{100 * v:.1f}' for v in m[side + '_offset_std'])}), distance {100 * m[side + '_offset_norm']:.1f} cm")
    shoulder_mid = lambda kp: kp[:, list(SHOULDERS)].mean(1) - kp[:, list(ROOT_JOINTS)].mean(1)   # noqa: E731
    e_neck = pose_distance(pred, gt, [NECK], "root")
    e_shm = np.linalg.norm(shoulder_mid(pred) - shoulder_mid(gt), axis=1)
    e_reloc = pose_distance(relocate_neck(pred, m["gt_offset"]), gt, [NECK], "root")
    m.update(neck_local=float(e_neck.mean()), neck_local_median=float(np.median(e_neck)),
             shoulder_mid_local=float(e_shm.mean()), neck_local_relocated=float(e_reloc.mean()),
             neck_local_relocated_median=float(np.median(e_reloc)))
    log(f"  root-aligned error: Neck {100 * m['neck_local']:.1f} cm (median {100 * m['neck_local_median']:.1f}), "
        f"shoulder midpoint {100 * m['shoulder_mid_local']:.1f} cm, Neck relocated to the GT offset "
        f"{100 * m['neck_local_relocated']:.1f} cm (median {100 * m['neck_local_relocated_median']:.1f})")
    return m


# --- visibility -------------------------------------------------------------------------------
def keypoint_similarity(pred2d, gt2d, area, sigmas=BODY25_SIGMAS):
    """Per-joint OKS terms exp(-d^2 / (2 s^2 k^2)) of the COCO keypoint similarity: pixel
    distance d between pred2d and gt2d (..., 25, 2), object scale s^2 = area (...,), k = 2 sigma
    per joint. The OKS is their mean over the labelled joints."""
    d2 = ((np.asarray(pred2d, float) - np.asarray(gt2d, float)) ** 2).sum(-1)
    return np.exp(-d2 / (2 * np.asarray(area, float)[..., None] * (2 * np.asarray(sigmas)) ** 2 + 1e-12))


def visibility_subset(gt, flags):
    """Copy of the COCO keypoint ground truth `gt` in which only the joints whose visibility
    flag is in `flags` stay labelled: the others get v = 0, as coco_eval's `exclude` does, and
    num_keypoints is recounted. Scored with sskit.coco.coco_eval, the keypoint (OKS) AP and the
    3D-LocSim AP are then taken over those joints only; humans without any such joint have
    num_keypoints 0 and are ignored by the COCO evaluation, like crowd regions."""
    anns = []
    for a in gt["annotations"]:
        kp = _kp2d(a)
        kp[~np.isin(kp[:, 2], list(flags)), 2] = 0
        anns.append(dict(a, keypoints=kp.ravel().tolist(), num_keypoints=int((kp[:, 2] > 0).sum())))
    return dict(gt, annotations=anns)


def _nanmean(x):
    return float(np.mean(x)) if len(x) else float("nan")


def eval_visibility(gt, dets, kp3d, log=print, iou_thr=0.5, midhip="hipmean", classes=VISIBILITY_CLASSES):
    """Errors of the IoU-matched pairs of eval_3d split by the GT visibility flag of each joint
    (`classes`: name -> flag; the SynLoc files use 2 visible, 1 occluded, 0 outside the image).
    Every joint of every matched person is scored once and the means are taken over the joints
    of a class, so the "all" entry reproduces the MPJPEs of eval_3d. The root and Procrustes
    alignments use all the scored joints as in eval_3d; only the averages are split. Returns
    dict(n, n_gt, n_pred, joints_averaged, classes, all=<stats>, <class>=<stats>, ...,
    people=dict(any_occluded, any_outside: fractions of the matched people, occluded_fraction:
    mean fraction of their scored joints that is occluded), orientation={axis: {visible: {n,
    angle_mean, heading_mean}, hidden: {...}}}: the orientation errors of the people whose axis
    joints are all visible against the others), where <stats> = dict(n_joints, fraction (of the
    scored joints), mpjpe_global, mpjpe_local, pa_mpjpe (metres), error_2d (pixels), oks_term
    (mean per-joint OKS term), per_joint_local, per_joint_global, per_joint_fraction (name ->
    value over the joints of that class, nan where there are none))."""
    use = joints_used(midhip)
    triples, (n_gt, n_pred) = matched_triples(gt, dets, kp3d, iou_thr)
    n = len(triples)
    m = dict(n=n, n_gt=n_gt, n_pred=n_pred, joints_averaged=[BODY25_NAMES[j] for j in use], classes=dict(classes))
    log(f"== visibility split (over the {n} matched people, {len(use)} joints each): errors of the joints flagged "
        + ", ".join(f"{name} (v = {flag})" for name, flag in classes.items()) + " in the ground truth")
    if n == 0:
        return m
    P, G = _skeletons(triples)
    V = np.array([_kp2d(a)[:, 2] for a, _, _ in triples])                              # (n, 25) flags
    g2, p2 = np.array([_kp2d(a)[:, :2] for a, _, _ in triples]), np.array([_kp2d(d)[:, :2] for _, d, _ in triples])
    area = np.array([float(a["area"]) for a, _, _ in triples])
    err = {"mpjpe_global": joint_errors(P, G, use, "none"),
           "mpjpe_local": joint_errors(P, G, use, "root"),
           "pa_mpjpe": joint_errors(P, G, use, "procrustes"),
           "error_2d": np.linalg.norm(p2 - g2, axis=-1),
           "oks_term": keypoint_similarity(p2, g2, area)}
    scored = np.zeros(25, bool)
    scored[use] = True
    masks = {"all": scored[None] & (V >= 0)}
    masks.update({name: scored[None] & (V == flag) for name, flag in classes.items()})
    n_scored = int(masks["all"].sum())
    log(f"  {'':<10} {'joints':>8} {'%':>6} {'global':>8} {'local':>8} {'PA':>8} {'2D px':>8} {'OKS term':>9}")
    for name, mask in masks.items():
        c = dict(n_joints=int(mask.sum()), fraction=float(mask.sum() / n_scored))
        for key, e in err.items():
            c[key] = _nanmean(e[mask])
        for key in ("mpjpe_local", "mpjpe_global"):
            c["per_joint_" + key.split("_")[1]] = {BODY25_NAMES[j]: _nanmean(err[key][mask[:, j], j]) for j in use}
        c["per_joint_fraction"] = {BODY25_NAMES[j]: float(mask[:, j].mean()) for j in use}
        m[name] = c
        log(f"  {name:<10} {c['n_joints']:8d} {100 * c['fraction']:6.1f} {100 * c['mpjpe_global']:8.1f} {100 * c['mpjpe_local']:8.1f} "
            f"{100 * c['pa_mpjpe']:8.1f} {c['error_2d']:8.1f} {c['oks_term']:9.3f}")
    occ, out = masks.get("occluded", np.zeros_like(V, bool)), masks.get("outside", np.zeros_like(V, bool))
    m["people"] = dict(any_occluded=float(occ.any(1).mean()), any_outside=float(out.any(1).mean()),
                       occluded_fraction=float((occ.sum(1) / masks["all"].sum(1)).mean()))
    log(f"  people: {100 * m['people']['any_occluded']:.1f} % have an occluded joint ({100 * m['people']['occluded_fraction']:.1f} % "
        f"of their joints on average), {100 * m['people']['any_outside']:.1f} % a joint outside the image")
    vis_flag = classes.get("visible", 2)
    oe = orientation_errors(P, G)
    m["orientation"] = {}
    for axis, (frm, to) in BODY25_ORIENTATION_AXES.items():
        both = (V[:, list(frm) + list(to)] == vis_flag).all(1)
        m["orientation"][axis] = {grp: dict(n=int(sel.sum()), angle_mean=_nanmean(oe[axis]["angle"][sel]),
                                            heading_mean=_nanmean(oe[axis]["heading"][sel]))
                                  for grp, sel in (("visible", both), ("hidden", ~both))}
        o = m["orientation"][axis]
        log(f"  {axis:<10} axis joints all visible: {o['visible']['n']:6d} people, angle {o['visible']['angle_mean']:5.1f} deg, "
            f"heading {o['visible']['heading_mean']:5.1f}; otherwise: {o['hidden']['n']:6d} people, angle {o['hidden']['angle_mean']:5.1f}, "
            f"heading {o['hidden']['heading_mean']:5.1f}")
    return m


# --- VoxelPose AP_K ---------------------------------------------------------------------------
def voxelpose_ap(dist, score, gt_id, n_gt, threshold):
    """VoxelPose's `_eval_list_to_ap` (lib/dataset/panoptic.py): predictions with distance
    `dist` to their ground-truth pose `gt_id` (-1: none) and confidence `score` are ranked
    by score; a prediction is a true positive if dist < threshold and its ground-truth pose
    has not been claimed by a higher-ranked prediction. Returns (AP, recall) with the VOC
    all-point interpolation of the precision over n_gt ground-truth poses."""
    dist, score, gt_id = np.asarray(dist, float), np.asarray(score, float), np.asarray(gt_id, int)
    order = np.argsort(-score, kind="stable")                 # list.sort(reverse=True) keeps ties in order too
    tp = np.zeros(len(order))
    claimed = set()
    for i, k in enumerate(order):
        if dist[k] < threshold and gt_id[k] >= 0 and gt_id[k] not in claimed:
            tp[i] = 1
            claimed.add(gt_id[k])
    fp = np.cumsum(1 - tp)
    tp = np.cumsum(tp)
    recall = tp / (n_gt + 1e-5)
    precision = tp / (tp + fp + 1e-5)
    for n in range(len(precision) - 2, -1, -1):
        precision[n] = max(precision[n], precision[n + 1])
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    index = np.where(recall[1:] != recall[:-1])[0]
    ap = float(np.sum((recall[index + 1] - recall[index]) * precision[index + 1]))
    return ap, float(recall[-2])


def joint_mask(anns, use, flags=None):
    """(n, 25) bool mask of the joints of the GT annotations `anns` that enter an average: the
    joints `use`, and, when `flags` is a set of COCO visibility flags, only those whose flag is
    in it (e.g. {2}: the visible joints)."""
    mask = np.zeros((len(anns), 25), bool)
    mask[:, use] = True
    if flags is not None:
        mask &= np.isin(np.array([_kp2d(a)[:, 2] for a in anns]).reshape(-1, 25), list(flags))
    return mask


def masked_distance(pred, gt, mask, align="none"):
    """MPJPE between pred (..., 25, 3) and gt (..., 25, 3) over the joints where mask (..., 25)
    is True (nan where it is empty); align as in `pose_distance` (none or root)."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    if align == "root":
        pred = pred - pred[..., ROOT_JOINTS, :].mean(-2, keepdims=True)
        gt = gt - gt[..., ROOT_JOINTS, :].mean(-2, keepdims=True)
    d = np.linalg.norm(pred - gt, axis=-1)
    n = mask.sum(-1)
    return np.where(n > 0, (d * mask).sum(-1) / np.maximum(n, 1), np.nan)


def pose_eval_list(gt, dets, kp3d, use, flags=None):
    """VoxelPose's eval list for `voxelpose_ap`: every detection is assigned the ground-truth
    pose of its image with the smallest MPJPE in world coordinates (VoxelPose's rule; images
    without ground truth give gt_id -1 and infinite distances). Returns (dist, score, gt_id,
    n_gt) with dist = {"global": MPJPE of the assigned pair, "root": its root-aligned
    MPJPE}, both over the joints `use`, restricted to the joints whose GT visibility flag is
    in `flags` when that is given (GT poses without any such joint are left out of n_gt and
    of the assignment)."""
    gt_by_img, det_by_img = by_image(gt, dets, kp3d)
    dist, score, gt_id, offset = {"global": [], "root": []}, [], [], {}
    n_gt = 0
    masks = {}
    for img_id, gts in gt_by_img.items():
        mask = joint_mask(gts, use, flags)
        keep = mask.any(1)
        gt_by_img[img_id], masks[img_id] = [a for a, k in zip(gts, keep) if k], mask[keep]
        offset[img_id] = n_gt
        n_gt += int(keep.sum())
    for img_id, ds in det_by_img.items():
        gts = gt_by_img.get(img_id, [])
        score += [float(d["score"]) for d, _ in ds]
        if not gts:
            for k in dist:
                dist[k] += [np.inf] * len(ds)
            gt_id += [-1] * len(ds)
            continue
        G = np.array([_kp3d(a) for a in gts])
        P = np.array([_kp3d(k) for _, k in ds])
        D = masked_distance(P[:, None], G[None], masks[img_id][None])         # (n_det, n_gt), world coordinates
        j = D.argmin(1)
        dist["global"] += D[np.arange(len(ds)), j].tolist()
        dist["root"] += masked_distance(P, G[j], masks[img_id][j], "root").tolist()
        gt_id += (offset[img_id] + j).tolist()
    return {k: np.array(v) for k, v in dist.items()}, np.array(score), np.array(gt_id, int), n_gt


def eval_ap_k(gt, dets, kp3d, log=print, thresholds=(0.05, 0.10, 0.15), midhip="hipmean", flags=None):
    """VoxelPose AP_K over all detections at each threshold K (metres), over the joints of
    eval_3d (MidHip excluded unless midhip="pelvis"): `global` with the MPJPE in world
    coordinates, the metric of the VoxelPose paper, and `root` with the root-aligned MPJPE
    of the same assigned pairs. With `flags`, a set of COCO visibility flags, only the GT
    joints with such a flag enter the MPJPEs (e.g. {2}: visible joints). Returns
    dict(thresholds, n_gt, n_pred, joints_averaged, [flags,] global={AP<K cm>, recall<K cm>,
    mAP}, root={...})."""
    use = joints_used(midhip)
    dist, score, gt_id, n_gt = pose_eval_list(gt, dets, kp3d, use, flags)
    m = dict(thresholds=[float(t) for t in thresholds], n_gt=n_gt, n_pred=len(score),
             joints_averaged=[BODY25_NAMES[j] for j in use])
    if flags is not None:
        m["flags"] = sorted(int(f) for f in flags)
    note = "" if flags is None else f" with visibility flag in {m['flags']}"
    log(f"== AP_K (VoxelPose): {len(score)} detections vs {n_gt} GT people, each detection assigned the GT pose "
        f"with the smallest MPJPE in world coordinates over the {len(use)} joints{note}; true positive if that MPJPE "
        f"(global) or the root-aligned MPJPE of the pair (root) is below K")
    for key, label in (("global", "global MPJPE (VoxelPose)"), ("root", "root-aligned MPJPE")):
        r, aps = {}, []
        for t in thresholds:
            ap, rec = voxelpose_ap(dist[key], score, gt_id, n_gt, t)
            k = f"{100 * t:g}"
            r[f"AP{k}"], r[f"recall{k}"] = ap, rec
            aps.append(ap)
        r["mAP"] = float(np.mean(aps)) if aps else float("nan")
        m[key] = r
        log(f"  {label + ':':<27}" + "   ".join(f"AP@{100 * t:g} cm {r[f'AP{100 * t:g}']:.3f} (recall {r[f'recall{100 * t:g}']:.3f})"
                                              for t in thresholds) + f"   mean {r['mAP']:.3f}")
    return m
