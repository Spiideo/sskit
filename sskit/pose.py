import numpy as np
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

ROOT_JOINTS = [9, 12]  # RHip, LHip: the root is their mean on both sides


def eval_3d(gt, dets, kp3d, iou_thr, log, midhip="hipmean"):
    from scipy.optimize import linear_sum_assignment

    use = [j for j in range(25) if midhip == "pelvis" or j != 8]   # joints entering the averages
    gt_by_img = {}
    for a in gt["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)
    det_by_img = {}
    for d, k in zip(dets, kp3d):
        det_by_img.setdefault(d["image_id"], []).append((d, k))
    cam_pos = {im["id"]: camera_position(im["camera_matrix"]).numpy()[:, 0] for im in gt["images"]}
    err_g, err_l, err_pa, root_err, depth_err, gt_depth = [], [], [], [], [], []
    n_gt = n_matched = 0
    for img_id, gts in gt_by_img.items():
        n_gt += len(gts)
        ds = det_by_img.get(img_id, [])
        if not ds:
            continue
        iou = box_iou(np.array([a["bbox"] for a in gts], float), np.array([d["bbox"] for d, _ in ds], float))
        rows, cols = linear_sum_assignment(-iou)
        cpos = cam_pos[img_id]
        for r, c in zip(rows, cols):
            if iou[r, c] < iou_thr:
                continue
            n_matched += 1
            g = np.array(gts[r]["keypoints_3d"], float).reshape(25, 4)[:, :3]
            p = np.array(ds[c][1]["keypoints_3d"], float)
            g_root, p_root = g[ROOT_JOINTS].mean(0), p[ROOT_JOINTS].mean(0)
            err_g.append(np.linalg.norm(p - g, axis=1))
            err_l.append(np.linalg.norm((p - p_root) - (g - g_root), axis=1))
            err_pa.append(np.linalg.norm(procrustes(p, g, use) - g, axis=1))
            root_err.append(p_root - g_root)
            g_dist, p_dist = np.linalg.norm(g_root - cpos), np.linalg.norm(p_root - cpos)
            depth_err.append(p_dist - g_dist)
            gt_depth.append(g_dist)
    err_g, err_l, err_pa = np.array(err_g), np.array(err_l), np.array(err_pa)
    root_err, depth_err, gt_depth = np.array(root_err), np.array(depth_err), np.array(gt_depth)
    m = dict(n_gt=n_gt, n_pred=len(dets), n_matched=n_matched, match_iou_thr=iou_thr, root="mean of RHip and LHip",
             midhip=midhip, joints_averaged=[BODY25_NAMES[j] for j in use])
    log(f"== 3D (BODY25, metres): {n_matched} of {n_gt} GT people matched to {len(dets)} predictions at IoU >= {iou_thr}; "
        f"root = hip mean, MidHip {'excluded from the averages' if midhip == 'hipmean' else 'scored against the GT pelvis'}")
    if n_matched == 0:
        return m
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
    fmt = lambda e: " ".join(f"{100 * v:6.1f}" if j in use else "     -" for j, v in enumerate(e))
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

