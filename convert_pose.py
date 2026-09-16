#!/usr/bin/env python
"""Create OpenPose BODY_25 poses for SoccerSceneV1 items.

Each item directory (``<scene>/<Camera>/``) holds ``objects.json`` with, per
human, the 10 SMPL-X betas (``smpl_shape``), the gender, the armature world
matrix (``smpl_matrix_world``) and the 55 SMPL-X joint positions in world
coordinates (``keypoints``). No pose parameters are stored, so the pose is
recovered here by inverse kinematics against the known joints:

1. targets are moved into the SMPL-X canonical frame of the armature
   (``inv(smpl_matrix_world @ RX90)``),
2. ``human_body_prior.IK_Engine`` (VPoser prior, LBFGS) fits root orientation,
   body pose and translation with the betas held fixed,
3. a short LBFGS refinement without the prior brings the joint residual to
   sub-millimetre level,
4. the gendered SMPL-X mesh is posed and the 11 OpenPose landmarks that are
   mesh vertices (nose, eyes, ears, toes, heels; ``smplx.vertex_ids``) are read
   off it and re-anchored to the exact joints from ``objects.json``,
5. BODY_25 is assembled with the smplify-x ``smpl_to_openpose`` mapping. The 14
   joint based points (Neck = SMPL-X ``neck``, MidHip = SMPL-X ``pelvis``, ...)
   are copied verbatim from ``objects.json``.

2D points are obtained with ``sskit.world_to_image`` followed by
``sskit.unnormalize``, i.e. the principal point is at ``((w-1)/2, (h-1)/2)``
(pixel centres). Note that the ``*_img`` keypoints in ``objects.json`` were
produced with an erroneous principal point at ``(w/2, h/2)`` and are therefore
0.7 px off compared to the values written here.

Outputs written next to ``objects.json``:

``openpose_body25.json``
    OpenPose JSON style: ``{"version": 1.3, "people": [{"person_id":
    [segmentation_id], "object_key": ..., "pose_keypoints_2d": [u, v, 1.0] *
    25, "pose_keypoints_3d": [x, y, z, 1.0] * 25, ...}]}``. Confidence is a
    constant 1.0.
``smplx_params.npz``
    Fitted SMPL-X parameters per human (``betas``, ``global_orient``,
    ``body_pose``, ``transl``, ``to_world`` ...) so meshes can be re-posed
    without refitting; see the ``readme`` entry inside the file.

Usage::

    python convert_pose.py BorasArenaCenterLeft2 --show body25.png
    python convert_pose.py --list SoccerSceneV1/val_v1.txt --batch-size 2048
    python convert_pose.py --list SoccerSceneV1/train_v1.txt --shard 0/4

Assets (SMPL-X v1.1 npz models and VPoser V02_05) are looked up in
``--models-dir`` / ``$SMPLX_ASSETS_DIR`` / ``~/.cache/smplx`` and can be
downloaded with ``fetch_smplx_assets.py`` (or ``--fetch``).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from sskit import imread, imshape, load_camera, unnormalize, world_to_image

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 55 SMPL-X joints in model order; identical to the keypoint names used in
# objects.json (verified against smplx.joint_names.JOINT_NAMES[:55]).
SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw",
    "left_eye_smplhf", "right_eye_smplhf",
    "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2",
    "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1",
    "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2",
    "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
    "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3",
]
try:  # prefer the authoritative list when smplx is importable
    from smplx.joint_names import JOINT_NAMES as _SMPLX_JOINT_NAMES

    assert list(_SMPLX_JOINT_NAMES[:55]) == SMPLX_JOINT_NAMES
except ImportError:  # pragma: no cover
    pass

NUM_JOINTS = 55
NUM_BETAS = 10

# Joints used as IK targets: body (0-21), jaw and eyes (fix head rotation),
# and the finger base joints (fix wrist rotation; their position does not
# depend on the finger pose, which is left at zero).
TARGET_IDX = list(range(25)) + [25, 28, 31, 34, 37, 40, 43, 46, 49, 52]

# Mesh vertices of the OpenPose landmarks (smplx.vertex_ids['smplx']), in the
# order smplx uses for joints 55-65.
LANDMARK_NAMES = ["nose", "reye", "leye", "rear", "lear",
                  "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
VERTEX_IDS = [9120, 9929, 9448, 616, 6, 5770, 5780, 8846, 8463, 8474, 8635]
try:
    from smplx.vertex_ids import vertex_ids as _vertex_ids

    assert [_vertex_ids["smplx"][n] for n in LANDMARK_NAMES] == VERTEX_IDS
except ImportError:  # pragma: no cover
    pass
# Joint whose exact position each landmark is re-anchored to: head for the
# face, foot joints for the toes and ankles for the heels.
LANDMARK_ANCHOR = [15, 15, 15, 15, 15, 10, 10, 7, 11, 11, 8]

# smplify-x smpl_to_openpose(model_type='smplx', openpose_format='coco25')
BODY25_FROM_SMPLX = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7,
                     56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
BODY25_NAMES = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
                "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
                "RBigToe", "RSmallToe", "RHeel"]
BODY25_PAIRS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10),
                (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16),
                (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]

# SMPL-X canonical frame (Y up, facing +Z) -> Blender armature frame (Z up).
RX90 = np.array([[1, 0, 0, 0],
                 [0, 0, -1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=np.float64)

OUTPUT_JSON = "openpose_body25.json"
OUTPUT_NPZ = "smplx_params.npz"
REQUIRED_FILES = ("objects.json", "camera_matrix.npy", "lens.json", "rgb.jpg")

FALLBACK_MODEL_DIRS = [
    Path("/home/hakan/src/human_body_prior/models/smplx"),
    Path("/home/hakan/src/smplx/smpl_models/smplx"),
]
FALLBACK_VPOSER_DIRS = [
    Path("/home/hakan/src/human_body_prior/support_data/dowloads/vposer_v2_05"),
]


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

def default_models_dir() -> Path:
    return Path(os.environ.get("SMPLX_ASSETS_DIR", Path.home() / ".cache" / "smplx"))


def resolve_assets(models_dir: Path, fetch: bool = False):
    """Return ({gender: npz path}, vposer_dir), downloading if requested."""
    def find():
        model_dirs = [models_dir / "models" / "smplx", models_dir / "smplx", models_dir] + FALLBACK_MODEL_DIRS
        vposer_dirs = [models_dir / "V02_05", models_dir / "vposer_v2_05"] + FALLBACK_VPOSER_DIRS
        models = None
        for d in model_dirs:
            paths = {g: d / f"SMPLX_{g.upper()}.npz" for g in ("male", "female", "neutral")}
            if all(p.exists() for p in paths.values()):
                models = paths
                break
        vposer = next((d for d in vposer_dirs if (d / "V02_05.yaml").exists()
                       and list((d / "snapshots").glob("*.ckpt"))), None)
        return models, vposer

    models, vposer = find()
    if fetch or models is None or vposer is None:
        sys.path.insert(0, str(HERE))
        import fetch_smplx_assets

        fetch_smplx_assets.main(["--dest", str(models_dir)])
        models, vposer = find()
    if models is None or vposer is None:
        sys.exit(f"SMPL-X models / VPoser not found under {models_dir}; run fetch_smplx_assets.py")
    return models, vposer


def import_hbp():
    """Import human_body_prior quietly (it prints about missing psbody.mesh)."""
    import warnings

    from loguru import logger

    logger.disable("human_body_prior")
    warnings.filterwarnings("ignore", category=FutureWarning, module="human_body_prior")
    with contextlib.redirect_stdout(io.StringIO()):
        from human_body_prior.body_model.body_model import BodyModel
        from human_body_prior.body_model.lbs import batch_rigid_transform, batch_rodrigues
        from human_body_prior.models.ik_engine import IK_Engine
    return BodyModel, IK_Engine, batch_rodrigues, batch_rigid_transform


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Human:
    item_dir: Path
    key: str
    segmentation_id: int
    gender: str
    betas: np.ndarray          # (10,)
    to_world: np.ndarray       # (4, 4) canonical SMPL-X frame -> world
    kp_world: np.ndarray       # (55, 3) exact joints from objects.json


def load_item_humans(item_dir: Path, objects: Optional[dict] = None) -> List[Human]:
    if objects is None:
        with open(item_dir / "objects.json") as fd:
            objects = json.load(fd)
    humans = []
    for key, obj in sorted(objects.items()):
        if obj.get("class") != "human":
            continue
        kp = obj.get("keypoints") or {}
        if not all(n in kp for n in SMPLX_JOINT_NAMES) or "smpl_shape" not in obj:
            continue
        gender = obj.get("metadata", {}).get("gender", "neutral").lower()
        if gender not in ("male", "female"):
            gender = "neutral"
        matrix_world = np.asarray(obj["smpl_matrix_world"], dtype=np.float64)
        to_world = matrix_world @ RX90
        if abs(np.linalg.det(to_world[:3, :3]) - 1) > 1e-4:
            raise ValueError(f"{item_dir}/{key}: smpl_matrix_world is not a rigid transform")
        humans.append(Human(
            item_dir=item_dir,
            key=key,
            segmentation_id=int(obj["segmentation_id"]),
            gender=gender,
            betas=np.asarray(obj["smpl_shape"], dtype=np.float64)[:NUM_BETAS],
            to_world=to_world,
            kp_world=np.array([kp[n] for n in SMPLX_JOINT_NAMES], dtype=np.float64),
        ))
    return humans


def to_canonical(humans: Sequence[Human]) -> np.ndarray:
    """World joints -> SMPL-X canonical frame of each armature, (B, 55, 3)."""
    out = np.empty((len(humans), NUM_JOINTS, 3))
    for i, h in enumerate(humans):
        inv = np.linalg.inv(h.to_world)
        out[i] = h.kp_world @ inv[:3, :3].T + inv[:3, 3]
    return out


def to_world(humans: Sequence[Human], pts: np.ndarray) -> np.ndarray:
    """(B, N, 3) canonical -> world using each human's to_world."""
    out = np.empty_like(pts)
    for i, h in enumerate(humans):
        out[i] = pts[i] @ h.to_world[:3, :3].T + h.to_world[:3, 3]
    return out


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class PoseFitter:
    """Recovers SMPL-X pose parameters from the 55 joint positions."""

    def __init__(self, model_paths: Dict[str, Path], vposer_dir: Path, device, ik_iter=300, refine_iter=100):
        BodyModel, IK_Engine, self._batch_rodrigues, self._batch_rigid_transform = import_hbp()
        self.device = device
        self.refine_iter = refine_iter
        self.bms = {g: BodyModel(str(p), num_betas=NUM_BETAS).to(device) for g, p in model_paths.items()}
        self.parents = next(iter(self.bms.values())).kintree_table[0].long()
        self.parents[0] = -1
        self.ik_engine = IK_Engine(
            vposer_expr_dir=str(vposer_dir),
            data_loss=lambda src, tgt: ((src - tgt) ** 2).sum(),
            optimizer_args={"type": "LBFGS", "max_iter": ik_iter, "lr": 1,
                            "tolerance_change": 1e-9, "history_size": 100},
            stepwise_weights=[{"data": 1e4, "poZ_body": 1e-2}],  # no 'betas': keep them fixed
            verbosity=0,
            num_betas=NUM_BETAS,
        ).to(device)
        self.ik_engine.logger = lambda *a, **k: None

    # -- kinematics -----------------------------------------------------
    def rest_joints(self, humans: Sequence[Human]) -> torch.Tensor:
        """Shaped joints in the zero pose, (B, 55, 3)."""
        out = torch.empty(len(humans), NUM_JOINTS, 3, device=self.device)
        for gender, bm in self.bms.items():
            idx = [i for i, h in enumerate(humans) if h.gender == gender]
            if idx:
                betas = torch.tensor(np.stack([humans[i].betas for i in idx]), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    out[idx] = bm(betas=betas).Jtr
        return out

    def joints_forward(self, J_rest, root_orient, pose_body, trans):
        B = J_rest.shape[0]
        full_pose = torch.cat([root_orient, pose_body, torch.zeros(B, 99, device=J_rest.device)], dim=-1)
        rot_mats = self._batch_rodrigues(full_pose.reshape(-1, 3)).reshape(B, NUM_JOINTS, 3, 3)
        posed, _ = self._batch_rigid_transform(rot_mats, J_rest, self.parents)
        return posed + trans[:, None]

    @staticmethod
    def init_root_and_trans(J_rest: torch.Tensor, targets: torch.Tensor):
        """Closed form: trans from the pelvis, root orientation by Kabsch on the pelvis children."""
        children = [1, 2, 3]
        r = (J_rest[:, children] - J_rest[:, :1]).double()
        d = (targets[:, children] - targets[:, :1]).double()
        H = r.transpose(1, 2) @ d
        U, _, Vh = torch.linalg.svd(H)
        det = torch.linalg.det(Vh.transpose(1, 2) @ U.transpose(1, 2))
        D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det.sign()], -1))
        R = Vh.transpose(1, 2) @ D @ U.transpose(1, 2)
        rotvec = Rotation.from_matrix(R.cpu().numpy()).as_rotvec()
        root_orient = torch.tensor(rotvec, dtype=torch.float32, device=J_rest.device)
        trans = targets[:, 0] - J_rest[:, 0]
        return root_orient, trans

    # -- stages ---------------------------------------------------------
    def fit(self, humans: Sequence[Human]):
        """Returns dict with root_orient, pose_body, trans (B,·) and residual stats in metres."""
        targets_np = to_canonical(humans)
        targets = torch.tensor(targets_np, dtype=torch.float32, device=self.device)
        betas = torch.tensor(np.stack([h.betas for h in humans]), dtype=torch.float32, device=self.device)
        J_rest = self.rest_joints(humans)
        root0, trans0 = self.init_root_and_trans(J_rest, targets)

        params = self.stage1_vposer(J_rest, targets, betas, root0, trans0)
        params, res = self.stage2_refine(J_rest, targets, params)

        bad = torch.isnan(res).any() or (res.amax(-1) > 0.05)
        if bad.any():
            # fall back to a prior-free fit from the closed form initialisation
            zero = {"root_orient": root0, "pose_body": torch.zeros_like(params["pose_body"]), "trans": trans0}
            params_z, res_z = self.stage2_refine(J_rest, targets, zero)
            better = torch.nan_to_num(res_z.mean(-1), nan=1e9) < torch.nan_to_num(res.mean(-1), nan=1e9)
            for k in params:
                params[k][better] = params_z[k][better]
            res[better] = res_z[better]
        params["J_rest"] = J_rest
        params["betas"] = betas
        params["targets"] = targets
        params["residual"] = res  # (B, len(TARGET_IDX)) metres
        return params

    def stage1_vposer(self, J_rest, targets, betas, root0, trans0):
        fitter = self

        class SourceKeyPoints(torch.nn.Module):
            kpts_colors = None
            bm_f = []

            def forward(self, body_parms):
                J = fitter.joints_forward(J_rest, body_parms["root_orient"], body_parms["pose_body"], body_parms["trans"])
                return {"source_kpts": J[:, TARGET_IDX], "body": SimpleNamespace(v=J)}

        init = {"betas": betas.clone(), "trans": trans0.clone(), "root_orient": root0.clone()}
        free = self.ik_engine(SourceKeyPoints().to(self.device), targets[:, TARGET_IDX], init)
        return {k: free[k].detach().clone() for k in ("root_orient", "pose_body", "trans")}

    def stage2_refine(self, J_rest, targets, init):
        vars_ = {k: torch.nn.Parameter(init[k].detach().clone()) for k in ("root_orient", "pose_body", "trans")}
        tgt = targets[:, TARGET_IDX]
        opt = torch.optim.LBFGS(list(vars_.values()), lr=1, max_iter=self.refine_iter, history_size=50,
                                tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            J = self.joints_forward(J_rest, vars_["root_orient"], vars_["pose_body"], vars_["trans"])
            loss = ((J[:, TARGET_IDX] - tgt) ** 2).sum()
            loss.backward()
            return loss

        for _ in range(3):
            opt.step(closure)
        params = {k: v.detach().clone() for k, v in vars_.items()}
        with torch.no_grad():
            J = self.joints_forward(J_rest, params["root_orient"], params["pose_body"], params["trans"])
            res = (J[:, TARGET_IDX] - tgt).norm(dim=-1)
        return params, res

    # -- landmarks --------------------------------------------------------
    def landmarks(self, humans: Sequence[Human], params):
        """Pose the gendered meshes, read the 11 landmark vertices and re-anchor them to the exact
        joints. Returns (landmarks_canonical (B,11,3), correction (B,11) metres)."""
        B = len(humans)
        lm = torch.empty(B, len(VERTEX_IDS), 3, device=self.device)
        corr = torch.empty(B, len(VERTEX_IDS), device=self.device)
        for gender, bm in self.bms.items():
            idx = [i for i, h in enumerate(humans) if h.gender == gender]
            if not idx:
                continue
            with torch.no_grad():
                body = bm(root_orient=params["root_orient"][idx], pose_body=params["pose_body"][idx],
                          betas=params["betas"][idx], trans=params["trans"][idx])
            v = body.v[:, VERTEX_IDS]
            shift = params["targets"][idx][:, LANDMARK_ANCHOR] - body.Jtr[:, LANDMARK_ANCHOR]
            lm[idx] = v + shift
            corr[idx] = shift.norm(dim=-1)
        return lm, corr


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def assemble_body25(kp_world: np.ndarray, lm_world: np.ndarray) -> np.ndarray:
    """(B,55,3) exact joints + (B,11,3) landmarks -> (B,25,3)."""
    return np.concatenate([kp_world, lm_world], axis=1)[:, BODY25_FROM_SMPLX]


def project(item_dir: Path, pts_world: np.ndarray) -> np.ndarray:
    """World (N,3) -> pixel (N,2) in rgb.jpg (sskit convention, principal point at ((w-1)/2, (h-1)/2))."""
    camera_matrix, dist_poly, _ = load_camera(item_dir)
    shape = imshape(item_dir / "rgb.jpg")
    pkt = torch.as_tensor(pts_world.reshape(-1, 3), dtype=torch.float32)
    uv = unnormalize(world_to_image(camera_matrix, dist_poly, pkt), shape)
    return uv.numpy().reshape(*pts_world.shape[:-1], 2)


def atomic_write(path: Path, write):
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)  # keep the suffix: np.savez appends .npz otherwise
    write(tmp)
    os.replace(tmp, path)


NPZ_README = """SMPL-X parameters fitted by convert_pose.py (sskit) to the 55 joint positions in objects.json.
Model: SMPL-X v1.1 gendered npz, num_betas=10, pose_hand/jaw/eye = 0 (flat hands).
Parameters are in the SMPL-X canonical frame (Y up); to_world (4x4) maps that frame to pitch/world
coordinates, i.e. world = to_world @ [smplx(betas, global_orient, body_pose, transl), 1].
to_world = smpl_matrix_world @ R_x(+90deg). residual_*_mm: joint fit error on the IK target joints.
landmark_correction_mm: shift applied to each of the 11 OpenPose landmark vertices to re-anchor them
to the exact joints (order: %s)."""


def write_outputs(item_dir: Path, humans: Sequence[Human], body25_3d: np.ndarray, body25_2d: np.ndarray,
                  params: dict, idx: Sequence[int], lm_corr: np.ndarray):
    people = []
    for j, h in enumerate(humans):
        kp2 = np.concatenate([body25_2d[j], np.ones((25, 1))], axis=1)
        kp3 = np.concatenate([body25_3d[j], np.ones((25, 1))], axis=1)
        people.append({
            "person_id": [h.segmentation_id],
            "object_key": h.key,
            "pose_keypoints_2d": [round(float(x), 3) for x in kp2.ravel()],
            "pose_keypoints_3d": [round(float(x), 5) for x in kp3.ravel()],
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d_frame": "world",
        })
    doc = {
        "version": 1.3,
        "keypoint_names": BODY25_NAMES,
        "note": "pose_keypoints_2d are pixels in rgb.jpg with the principal point at ((w-1)/2, (h-1)/2) "
                "(sskit.unnormalize; the objects.json *_img keypoints used (w/2, h/2) and are 0.7 px off); "
                "pose_keypoints_3d are world/pitch metres; confidence is constant 1.0; "
                "Neck/MidHip are the SMPL-X neck/pelvis joints (smplify-x convention).",
        "people": people,
    }
    atomic_write(item_dir / OUTPUT_JSON, lambda p: p.write_text(json.dumps(doc)))

    cpu = {k: params[k][idx].cpu().numpy() for k in ("root_orient", "pose_body", "trans", "betas", "residual")}
    arrays = dict(
        object_key=np.array([h.key for h in humans]),
        segmentation_id=np.array([h.segmentation_id for h in humans], dtype=np.int64),
        gender=np.array([h.gender for h in humans]),
        betas=cpu["betas"].astype(np.float32),
        global_orient=cpu["root_orient"].astype(np.float32),
        body_pose=cpu["pose_body"].astype(np.float32),
        transl=cpu["trans"].astype(np.float32),
        to_world=np.stack([h.to_world for h in humans]).astype(np.float64),
        residual_mean_mm=(cpu["residual"].mean(-1) * 1000).astype(np.float32),
        residual_max_mm=(cpu["residual"].max(-1) * 1000).astype(np.float32),
        landmark_correction_mm=(lm_corr * 1000).astype(np.float32),
        body25_3d=body25_3d.astype(np.float64),
        body25_2d=body25_2d.astype(np.float64),
        readme=np.array(NPZ_README % ", ".join(LANDMARK_NAMES)),
    )
    atomic_write(item_dir / OUTPUT_NPZ, lambda p: np.savez_compressed(p, **arrays))


def draw_show(item_dir: Path, body25_2d: np.ndarray, out: Path):
    """Draw the BODY_25 skeletons on rgb.jpg: joints red, face landmarks yellow, feet blue."""
    from PIL import ImageDraw
    from torchvision.transforms.functional import to_pil_image

    img = to_pil_image(imread(str(item_dir / "rgb.jpg")))
    draw = ImageDraw.Draw(img)
    colors = ["red"] * 25
    for i in (0, 15, 16, 17, 18):
        colors[i] = "yellow"
    for i in range(19, 25):
        colors[i] = "deepskyblue"
    for kp in body25_2d:
        for a, b in BODY25_PAIRS:
            draw.line([tuple(kp[a]), tuple(kp[b])], fill="lime", width=2)
        for i, (u, v) in enumerate(kp[[22,19]]):
            r = 1
            draw.ellipse([u - r, v - r, u + r, v + r], fill=colors[i])
    img.save(str(out))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def iter_items(args) -> List[Path]:
    items = [Path(d) for d in args.item_dirs]
    for lst in args.list:
        lst = Path(lst)
        root = Path(args.root) if args.root else lst.parent
        for line in lst.read_text().splitlines():
            line = line.strip()
            if line:
                p = root / line
                items.append(p.parent if p.suffix else p)
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        items = items[i::n]
    return items


def needs_processing(item_dir: Path, overwrite: bool) -> bool:
    return overwrite or not ((item_dir / OUTPUT_JSON).exists() and (item_dir / OUTPUT_NPZ).exists())


def process_group(fitter: PoseFitter, group: List[List[Human]], args, stats: dict):
    humans = [h for hs in group for h in hs]
    if humans:
        t0 = time.time()
        params = fitter.fit(humans)
        lm, corr = fitter.landmarks(humans, params)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        stats["fit_time"] += time.time() - t0
        stats["humans"] += len(humans)
        res = params["residual"]
        stats["res_mean_sum"] += float(res.mean(-1).sum())
        stats["res_max"] = max(stats["res_max"], float(res.max()))
        stats["corr_max"] = max(stats["corr_max"], float(corr.max()))
        lm_world_all = to_world(humans, lm.cpu().numpy().astype(np.float64))
        corr_all = corr.cpu().numpy()
    start = 0
    for hs in group:
        item_dir = hs[0].item_dir if hs else None
        idx = list(range(start, start + len(hs)))
        start += len(hs)
        if not hs:
            continue
        kp_world = np.stack([h.kp_world for h in hs])
        body25_3d = assemble_body25(kp_world, lm_world_all[idx])
        body25_2d = project(item_dir, body25_3d)
        write_outputs(item_dir, hs, body25_3d, body25_2d, params, idx, corr_all[idx])
        if args.show:
            draw_show(item_dir, body25_2d, Path(args.show))
        stats["items"] += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("item_dirs", nargs="*", help="item directories containing objects.json")
    parser.add_argument("--list", action="append", default=[], help="split list file (lines like ./scene/Camera/rgb.jpg); repeatable")
    parser.add_argument("--root", help="dataset root for --list entries (default: the list file's directory)")
    parser.add_argument("--shard", help="I/N: process every N-th item starting at I")
    parser.add_argument("--batch-size", type=int, default=2048, help="humans fitted per optimisation batch (cost is per LBFGS iteration, so bigger is faster; ~1 GB GPU memory at 2048)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models-dir", type=Path, default=default_models_dir(), help="SMPL-X assets dir ($SMPLX_ASSETS_DIR or ~/.cache/smplx)")
    parser.add_argument("--fetch", action="store_true", help="download assets with fetch_smplx_assets.py")
    parser.add_argument("--overwrite", action="store_true", help="recompute items that already have outputs")
    parser.add_argument("--ik-iter", type=int, default=300, help="LBFGS iterations for the VPoser stage")
    parser.add_argument("--refine-iter", type=int, default=100, help="LBFGS iterations per refinement step")
    parser.add_argument("--show", help="write a debug PNG with the BODY_25 skeletons drawn on rgb.jpg (single item)")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be processed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    items = iter_items(args)
    if not items:
        parser.error("no items given; pass item directories or --list")
    todo, skipped, missing = [], 0, 0
    for item in items:
        absent = [f for f in REQUIRED_FILES if not (item / f).exists()]
        if absent:
            missing += 1
            if args.verbose:
                print(f"missing {', '.join(absent)}: {item}", file=sys.stderr)
        elif needs_processing(item, args.overwrite):
            todo.append(item)
        else:
            skipped += 1
    print(f"{len(todo)} items to process, {skipped} already done, {missing} missing or incomplete", file=sys.stderr)
    if args.dry_run or not todo:
        return

    model_paths, vposer_dir = resolve_assets(args.models_dir, fetch=args.fetch)
    fitter = PoseFitter(model_paths, vposer_dir, torch.device(args.device), args.ik_iter, args.refine_iter)

    stats = dict(items=0, humans=0, fit_time=0.0, res_mean_sum=0.0, res_max=0.0, corr_max=0.0)
    group, n_group = [], 0
    for item in tqdm(todo, unit="item", disable=not sys.stderr.isatty()):
        try:
            humans = load_item_humans(item)
        except (OSError, ValueError, KeyError) as e:
            print(f"error reading {item}: {e}", file=sys.stderr)
            continue
        group.append(humans)
        n_group += len(humans)
        if n_group >= args.batch_size:
            process_group(fitter, group, args, stats)
            group, n_group = [], 0
    if group:
        process_group(fitter, group, args, stats)

    if stats["humans"]:
        print(f"processed {stats['items']} items, {stats['humans']} humans in {stats['fit_time']:.1f} s "
              f"({stats['humans'] / max(stats['fit_time'], 1e-9):.1f} humans/s); "
              f"joint residual mean {1000 * stats['res_mean_sum'] / stats['humans']:.3f} mm, "
              f"max {1000 * stats['res_max']:.3f} mm; max landmark re-anchoring {1000 * stats['corr_max']:.3f} mm",
              file=sys.stderr)


if __name__ == "__main__":
    main()
