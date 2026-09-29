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

Scenes may also be stored as ``<scene>.tar.bz2`` archives (members
``./<Camera>/...``). When an item directory does not exist but the scene
archive does, the archive is used: bz2 tars have no index, so all archives of
the run are decompressed up front (``--threads`` in parallel) and the small
members the script needs are kept in memory, zlib compressed, for the whole
run; ``rgb.jpg`` only contributes its size. Outputs are added to the archives,
which are rewritten atomically after each fitting batch (in parallel;
concurrent ``--shard`` runs serialise on a ``.convert_pose.lock`` file next to
the archives). An archive can also be given directly as an item to process all
its cameras.

Outputs written next to ``objects.json`` (added to the archive for archived scenes):

``openpose_body25.json``
    OpenPose JSON style: ``{"version": 1.3, "people": [{"person_id":
    [segmentation_id], "object_key": ..., "pose_keypoints_2d": [u, v, 1.0] *
    25, "pose_keypoints_3d": [x, y, z, 1.0] * 25, ...}]}``. Confidence is a
    constant 1.0.
``smplx_params.npz``
    Fitted SMPL-X parameters per human (``betas``, ``global_orient``,
    ``body_pose``, ``transl``, ``to_world`` ...) so meshes can be re-posed
    without refitting; see the ``readme`` entry inside the file.

With ``--coco-out`` a collected COCO keypoint style file is written in
addition (requires a single ``--list``). It reuses the ``images`` of the
existing SynLoc bbox annotation file for the same split (``mini.json`` next to
the output, or ``--coco-images``; list line ``i`` is image ``i``, verified via
the camera matrix) and, per human, the ``id``, ``bbox``, ``area`` and
``position_on_pitch`` of the matching bbox annotation, adding flat COCO
``keypoints`` (25 x ``[u, v, v_flag]``), ``keypoints_3d`` (25 x ``[x, y, z, 1]``,
world metres) and ``num_keypoints``. Keypoints projecting outside the image
get ``v_flag = 0`` (and ``u = v = 0``), the others ``v_flag = 2``; humans with
none of the 15 FIFA joints (``sskit.pose.BODY25_TO_FIFA15``) inside the image
are dropped. The category carries the BODY_25 keypoint names and skeleton.

Usage::

    python convert_pose.py BorasArenaCenterLeft2 --show body25.png
    python convert_pose.py SoccerSceneV1/20231226_153701_node041_10100346_005_003.tar.bz2
    python convert_pose.py --list SoccerSceneV1/val_v1.txt --batch-size 2048
    python convert_pose.py --list SoccerSceneV1/train_v1.txt --shard 0/4
    python convert_pose.py --list SoccerSceneV1/mini_v2.txt --coco-out SoccerNet/SpiideoSynLoc/annotations/person_keypoints_mini.json

Assets (SMPL-X v1.1 npz models and VPoser V02_05) are looked up in
``--models-dir`` / ``$SMPLX_ASSETS_DIR`` / ``~/.cache/smplx`` and should contain::

    ./V02_05
    ./V02_05/V02_05.log
    ./V02_05/V02_05.yaml
    ./V02_05/snapshots
    ./V02_05/snapshots/V02_05_epoch=13_val_loss=0.03.ckpt
    ./zips
    ./models
    ./models/smplx
    ./models/smplx/SMPLX_NEUTRAL.pkl
    ./models/smplx/SMPLX_NEUTRAL.npz
    ./models/smplx/SMPLX_MALE.npz
    ./models/smplx/version.txt
    ./models/smplx/SMPLX_FEMALE.npz
    ./models/smplx/smplx_npz.zip
    ./models/smplx/SMPLX_FEMALE.pkl
    ./models/smplx/SMPLX_MALE.pkl
"""
import argparse
import collections
import contextlib
import fcntl
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from sskit import imshape, make_camera, unnormalize, world_to_image
from sskit.pose import BODY25_TO_FIFA15
from smplx.vertex_ids import vertex_ids as _vertex_ids
from smplx.joint_names import JOINT_NAMES as _smplx_joint_names


HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_JOINTS = 55
NUM_BETAS = 10

# The 55 SMPL-X joints in model order; identical to the keypoint names used in
# objects.json
SMPLX_JOINT_NAMES = list(_smplx_joint_names[:NUM_JOINTS])

# Joints used as IK targets: body (0-21), jaw and eyes (fix head rotation),
# and the finger base joints (fix wrist rotation; their position does not
# depend on the finger pose, which is left at zero).
TARGET_IDX = list(range(25)) + [25, 28, 31, 34, 37, 40, 43, 46, 49, 52]

# Mesh vertices of the OpenPose landmarks (smplx.vertex_ids['smplx']), in the
# order smplx uses for joints 55-65.
LANDMARK_NAMES = ["nose", "reye", "leye", "rear", "lear",
                  "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
VERTEX_IDS = [_vertex_ids["smplx"][n] for n in LANDMARK_NAMES]

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
                (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (11, 23), (11, 24)]

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
# Items: directories or members of <scene>.tar.bz2 archives
# ---------------------------------------------------------------------------

ARCHIVE_SUFFIX = ".tar.bz2"
LOCK_FILE = ".convert_pose.lock"
# Members kept in memory (zlib compressed) for every archive of the run; other members are
# read by rescanning the archive when needed (segmentations for unmatched humans, --show).
CACHED_MEMBERS = {"objects.json", "camera_matrix.npy", "lens.json", "areas_cache.json", OUTPUT_JSON}
IMAGE_MEMBER = "rgb.jpg"


def default_threads() -> int:
    try:
        return min(32, len(os.sched_getaffinity(0)))
    except AttributeError:
        return min(32, os.cpu_count() or 1)


@contextlib.contextmanager
def file_lock(path: Path):
    """Exclusive flock on `path` (created if missing); degrades to no locking with a warning."""
    fd = None
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as e:
        print(f"warning: cannot lock {path} ({e}); concurrent archive updates are unsafe", file=sys.stderr)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)  # releases the lock


class Archive:
    """One <scene>.tar.bz2: the names of all members, the CACHED_MEMBERS contents, the image sizes,
    and files to add on flush(). Reading it decompresses the whole archive once."""

    def __init__(self, path: Path):
        self.path = path
        self.prefix = ""                       # "./" if the members are stored that way
        self.names: set = set()                # regular file members, without the prefix
        self.data: Dict[str, bytes] = {}       # zlib compressed member contents
        self.shapes: Dict[str, tuple] = {}     # (3, h, w) of the image members
        self.pending: Dict[str, bytes] = {}
        self.owner = (os.getuid(), os.getgid(), "", "")
        with tarfile.open(path, "r|bz2") as tar:
            for m in tar:
                if m.name.startswith("./"):
                    self.prefix = "./"
                if not m.isfile():
                    continue
                name = self.strip(m.name)
                self.names.add(name)
                self.owner = (m.uid, m.gid, m.uname, m.gname)
                base = name.rsplit("/", 1)[-1]
                if base in CACHED_MEMBERS:
                    self.data[name] = zlib.compress(tar.extractfile(m).read(), 1)
                elif base == IMAGE_MEMBER:
                    self.shapes[name] = imshape(io.BytesIO(tar.extractfile(m).read()))

    @staticmethod
    def strip(name: str) -> str:
        return name[2:] if name.startswith("./") else name

    def exists(self, name: str) -> bool:
        return name in self.names

    def read(self, name: str) -> bytes:
        if name in self.data:
            return zlib.decompress(self.data[name])
        if name not in self.names:
            raise FileNotFoundError(f"{self.path}:{name}")
        with tarfile.open(self.path, "r|bz2") as tar:  # uncached member: rescan
            for m in tar:
                if m.isfile() and self.strip(m.name) == name:
                    return tar.extractfile(m).read()
        raise FileNotFoundError(f"{self.path}:{name}")

    def image_shape(self, name: str) -> tuple:
        return self.shapes[name]

    def write(self, name: str, data: bytes):
        self.pending[name] = data
        self.names.add(name)
        if name.rsplit("/", 1)[-1] in CACHED_MEMBERS:
            self.data[name] = zlib.compress(data, 1)

    def flush(self):
        """Rewrite the archive with the pending files added (replacing members of the same name).
        The caller holds the lock of the archive's directory."""
        if not self.pending:
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tarfile.open(self.path, "r|bz2") as old, tarfile.open(tmp, "w:bz2") as new:
            for m in old:
                if self.strip(m.name) in self.pending:
                    continue
                new.addfile(m, old.extractfile(m) if m.isfile() else None)
            for name, data in self.pending.items():
                info = tarfile.TarInfo(self.prefix + name)
                info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
                info.uid, info.gid, info.uname, info.gname = self.owner
                new.addfile(info, io.BytesIO(data))
        os.replace(tmp, self.path)
        self.pending.clear()


class ArchiveCache:
    """All archives touched by the run. Loading and rewriting run in a thread pool (bz2 releases the GIL)."""

    def __init__(self, threads: Optional[int] = None):
        self.threads = threads or default_threads()
        self.archives: Dict[Path, Archive] = {}
        self.show_count = 0

    def get(self, path: Path) -> Archive:
        if path not in self.archives:
            self.archives[path] = Archive(path)
        return self.archives[path]

    def prefetch(self, paths: Sequence[Path]):
        todo = list(dict.fromkeys(p for p in paths if p not in self.archives))
        if not todo:
            return
        with ThreadPoolExecutor(self.threads) as pool:
            for path, archive in zip(todo, tqdm(pool.map(Archive, todo), total=len(todo), unit="archive",
                                                desc="reading archives", disable=not sys.stderr.isatty())):
                self.archives[path] = archive

    def flush(self):
        """Rewrite all archives with pending files, holding the lock of each directory involved."""
        dirty = [a for a in self.archives.values() if a.pending]
        if not dirty:
            return
        with contextlib.ExitStack() as stack:
            for directory in sorted({a.path.parent for a in dirty}):
                stack.enter_context(file_lock(directory / LOCK_FILE))
            with ThreadPoolExecutor(self.threads) as pool:
                list(pool.map(Archive.flush, dirty))


ARCHIVES = ArchiveCache()


class Item:
    """An item ``<scene>/<Camera>``: a directory, or (when `archive` is given) the member directory
    ``<Camera>`` of ``<scene>.tar.bz2``. Files are read into memory and written back atomically."""

    def __init__(self, path: Path, archive: Optional[Path] = None, member: Optional[str] = None):
        self.path = path
        self.archive = archive
        self.member = (member or path.name) if archive else None   # member directory inside the archive

    def __str__(self):
        return f"{self.archive}:{self.member}" if self.archive else str(self.path)

    def _member(self, name: str) -> str:
        return f"{self.member}/{name}"

    def exists(self, name: str) -> bool:
        if self.archive:
            return ARCHIVES.get(self.archive).exists(self._member(name))
        return (self.path / name).exists()

    def read_bytes(self, name: str) -> bytes:
        if self.archive:
            return ARCHIVES.get(self.archive).read(self._member(name))
        return (self.path / name).read_bytes()

    def read_json(self, name: str):
        return json.loads(self.read_bytes(name))

    def image_shape(self, name: str = IMAGE_MEMBER) -> tuple:
        """(3, h, w) of an image member, without reading the pixels."""
        if self.archive:
            return ARCHIVES.get(self.archive).image_shape(self._member(name))
        return imshape(self.path / name)

    def write(self, name: str, data: bytes):
        if self.archive:
            ARCHIVES.get(self.archive).write(self._member(name), data)
        else:
            atomic_write(self.path / name, lambda p: p.write_bytes(data))


def make_item(path: Path) -> Item:
    """Item for ``<root>/<scene>/[<sub>/]<Camera>``; if the directory is absent, the closest ancestor with
    a ``<ancestor>.tar.bz2`` next to it is the archive and the rest of the path the member directory."""
    if not path.is_dir():
        for ancestor in path.parents:
            if not ancestor.name:
                break
            archive = ancestor.with_name(ancestor.name + ARCHIVE_SUFFIX)
            if archive.is_file():
                return Item(path, archive, path.relative_to(ancestor).as_posix())
    return Item(path)


def archive_items(archive: Path) -> List[Item]:
    """All camera items (member directories with an objects.json) of a scene archive."""
    scene = archive.with_name(archive.name[:-len(ARCHIVE_SUFFIX)])
    cameras = sorted(n.rsplit("/", 1)[0] for n in ARCHIVES.get(archive).names if n.endswith("/objects.json"))
    return [Item(scene / camera, archive, camera) for camera in cameras]


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

def default_models_dir() -> Path:
    return Path(os.environ.get("SMPLX_ASSETS_DIR", Path.home() / ".cache" / "smplx"))


def resolve_assets(models_dir: Path):
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
    if models is None or vposer is None:
        sys.exit(f"SMPL-X models / VPoser not found under {models_dir}")
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
    item: Item
    key: str
    segmentation_id: int
    gender: str
    betas: np.ndarray          # (10,)
    to_world: np.ndarray       # (4, 4) canonical SMPL-X frame -> world
    kp_world: np.ndarray       # (55, 3) exact joints from objects.json


def load_item_humans(item: Item, objects: Optional[dict] = None) -> List[Human]:
    if objects is None:
        objects = item.read_json("objects.json")
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
            raise ValueError(f"{item}/{key}: smpl_matrix_world is not a rigid transform")
        humans.append(Human(
            item=item,
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

        bad = torch.isnan(res).any(-1) | (res.amax(-1) > 0.05)
        if bad.any():
            # fall back to a prior-free fit from the closed form initialisation if it is better
            zero = {"root_orient": root0, "pose_body": torch.zeros_like(params["pose_body"]), "trans": trans0}
            params_z, res_z = self.stage2_refine(J_rest, targets, zero)
            better = torch.nan_to_num(res_z.mean(-1), nan=1e9) < torch.nan_to_num(res.mean(-1), nan=1e9)
            better &= bad
            for k in params:
                params[k][better] = params_z[k][better]
            res[better] = res_z[better]
        assert not torch.isnan(res).any()

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


def project(item: Item, pts_world: np.ndarray) -> np.ndarray:
    """World (N,3) -> pixel (N,2) in rgb.jpg (sskit convention, principal point at ((w-1)/2, (h-1)/2))."""
    camera_matrix, dist_poly, _ = make_camera(np.load(io.BytesIO(item.read_bytes("camera_matrix.npy"))),
                                              item.read_json("lens.json"))
    shape = item.image_shape()
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


def write_outputs(item: Item, humans: Sequence[Human], body25_3d: np.ndarray, body25_2d: np.ndarray,
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
    item.write(OUTPUT_JSON, json.dumps(doc).encode())

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
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    item.write(OUTPUT_NPZ, buf.getvalue())


def draw_show(item: Item, body25_2d: np.ndarray, out: Path):
    """Draw the BODY_25 skeletons on rgb.jpg: joints red, face landmarks yellow, feet blue."""
    from PIL import Image, ImageDraw

    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(item.read_bytes("rgb.jpg"))).convert("RGB")
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
# Collected COCO keypoint file
# ---------------------------------------------------------------------------

def default_coco_images(coco_out: Path, list_file: Path) -> Path:
    """SoccerNet/SpiideoSynLoc/annotations/<split>.json for a list file <split>_vN.txt."""
    split = re.sub(r"_v\d+$", "", list_file.stem)
    return coco_out.parent / f"{split}.json"


def object_area(item: Item, key: str, obj: dict, seg_cache: dict) -> int:
    """Pixel area of a human, from areas_cache.json (written by make_coco.py) or the segmentation."""
    if "areas" not in seg_cache:
        seg_cache["areas"] = item.read_json("areas_cache.json") if item.exists("areas_cache.json") else {}
    if key not in seg_cache["areas"]:
        if "seg" not in seg_cache:
            seg_cache["seg"] = np.load(io.BytesIO(gzip.decompress(item.read_bytes("segmentations.npy.gz"))))
        seg_cache["areas"][key] = int((seg_cache["seg"] == obj["segmentation_id"]).sum())
    return seg_cache["areas"][key]


def write_coco(coco_out: Path, coco_images: Path, list_file: Path, items: List[Item]):
    """Collect the per-item openpose_body25.json files of `items` (in list order) into one COCO
    keypoint style file that references the images of the bbox annotation file `coco_images`."""
    ref = json.loads(coco_images.read_text())
    if len(ref["images"]) != len(items):
        raise ValueError(f"{coco_images} has {len(ref['images'])} images but {list_file} has {len(items)} lines")
    ref_anns: Dict[int, list] = {}
    for a in ref["annotations"]:
        ref_anns.setdefault(a["image_id"], []).append(a)

    annotations, next_id = [], max((a["id"] for a in ref["annotations"]), default=-1) + 1
    n_missing = n_unmatched = n_outside = 0
    for image, item in zip(ref["images"], items):
        if not item.exists(OUTPUT_JSON):
            n_missing += 1
            continue
        camera_matrix = np.load(io.BytesIO(item.read_bytes("camera_matrix.npy")))[:3]
        if not np.allclose(camera_matrix, np.array(image["camera_matrix"]), atol=1e-5):
            raise ValueError(f"camera matrix of image {image['id']} in {coco_images} does not match {item}; "
                             "is the list file the one the bbox annotations were made from?")
        objects = item.read_json("objects.json")
        people = item.read_json(OUTPUT_JSON)["people"]
        candidates = list(ref_anns.get(image["id"], []))
        seg_cache: dict = {}
        _, height, width = item.image_shape()
        for person in people:
            kp2 = np.array(person["pose_keypoints_2d"]).reshape(25, 3)
            kp3 = np.array(person["pose_keypoints_3d"]).reshape(25, 4)
            # COCO visibility flag: 2 = labelled and visible (occlusion is not evaluated), 0 = outside the image
            inside = ((kp2[:, 0] >= 0) & (kp2[:, 0] <= width-1) & (kp2[:, 1] >= 0) & (kp2[:, 1] <= height-1))
            if not inside[BODY25_TO_FIFA15].any():
                n_outside += 1
                continue
            kp2[:, 2] = np.where(inside, 2, 0)
            kp2[~inside, :2] = 0
            obj = objects[person["object_key"]]
            pelvis = np.array(obj["keypoints"]["pelvis"])
            match = next((a for a in candidates if np.allclose(a["keypoints_3d"][0][:3], pelvis, atol=1e-6)), None)
            if match is not None:
                candidates.remove(match)
                ann = {k: match[k] for k in ("id", "position_on_pitch", "bbox", "area") if k in match}
            else:
                n_unmatched += 1
                u0, u1, v0, v1 = obj.get("bounding_box_tighter", obj["bounding_box_tight"])
                ann = dict(id=next_id, position_on_pitch=[float(pelvis[0]), float(pelvis[1])],
                           bbox=[u0, v0, u1 - u0, v1 - v0], area=object_area(item, person["object_key"], obj, seg_cache))
                next_id += 1
            ann.update(
                image_id=image["id"],
                category_id=1,
                iscrowd=0,
                segmentation_id=person["person_id"][0],
                keypoints=[round(float(x), 3) for x in kp2.ravel()],
                keypoints_3d=[round(float(x), 5) for x in kp3.ravel()],
                num_keypoints=int(inside.sum()),
            )
            annotations.append(ann)

    doc = dict(
        images=ref["images"],
        annotations=annotations,
        categories=[dict(id=1, name="person", supercategory="person", keypoints=BODY25_NAMES,
                         skeleton=[[a + 1, b + 1] for a, b in BODY25_PAIRS])],
        info=dict(description="OpenPose BODY_25 keypoints for Spiideo SoccerNet SynLoc, generated by sskit/convert_pose.py "
                              "from SMPL-X fits; keypoints are pixels (principal point ((w-1)/2, (h-1)/2)), keypoints_3d are "
                              "world/pitch metres with a trailing 1; keypoints outside the image have visibility 0 and u = v = 0; "
                              "humans with none of the 15 FIFA joints inside the image are omitted; "
                              "MidHip/Neck are the SMPL-X pelvis/neck joints.",
                  source_list=str(list_file), bbox_annotations=str(coco_images)),
    )
    coco_out.parent.mkdir(parents=True, exist_ok=True)

    def dump(path):
        with open(path, "w") as fd:
            json.dump(doc, fd)

    atomic_write(coco_out, dump)
    print(f"wrote {coco_out}: {len(doc['images'])} images, {len(annotations)} annotations "
          f"({n_missing} items without pose output, {n_unmatched} humans without bbox annotation, "
          f"{n_outside} humans dropped with no FIFA joint inside the image)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def iter_items(args) -> List[Item]:
    items = []
    for d in args.item_dirs:
        p = Path(d)
        if p.name.endswith(ARCHIVE_SUFFIX) and p.is_file():
            items += archive_items(p)
        else:
            items.append(make_item(p))
    for lst in args.list:
        lst = Path(lst)
        root = Path(args.root) if args.root else lst.parent
        for line in lst.read_text().splitlines():
            line = line.strip()
            if line:
                p = root / line
                items.append(make_item(p.parent if p.suffix else p))
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        items = items[i::n]
    return items


def needs_processing(item: Item, overwrite: bool) -> bool:
    return overwrite or not (item.exists(OUTPUT_JSON) and item.exists(OUTPUT_NPZ))


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
        item = hs[0].item if hs else None
        idx = list(range(start, start + len(hs)))
        start += len(hs)
        if not hs:
            continue
        kp_world = np.stack([h.kp_world for h in hs])
        body25_3d = assemble_body25(kp_world, lm_world_all[idx])
        body25_2d = project(item, body25_3d)
        write_outputs(item, hs, body25_3d, body25_2d, params, idx, corr_all[idx])
        if args.show:
            draw_show(item, body25_2d, Path(args.show) / f'{ARCHIVES.show_count:06d}.png')
            ARCHIVES.show_count += 1
        stats["items"] += 1

    ARCHIVES.flush()  # rewrite the touched scene archives once per group


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("item_dirs", nargs="*", help="item directories containing objects.json, or <scene>.tar.bz2 archives")
    parser.add_argument("--list", action="append", default=[], help="split list file (lines like ./scene/Camera/rgb.jpg); repeatable")
    parser.add_argument("--root", help="dataset root for --list entries (default: the list file's directory)")
    parser.add_argument("--shard", help="I/N: process every N-th item starting at I")
    parser.add_argument("--batch-size", type=int, default=2048, help="humans fitted per optimisation batch (cost is per LBFGS iteration, so bigger is faster; ~1 GB GPU memory at 2048)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models-dir", type=Path, default=default_models_dir(), help="SMPL-X assets dir ($SMPLX_ASSETS_DIR or ~/.cache/smplx)")
    parser.add_argument("--overwrite", action="store_true", help="recompute items that already have outputs")
    parser.add_argument("--ik-iter", type=int, default=300, help="LBFGS iterations for the VPoser stage")
    parser.add_argument("--refine-iter", type=int, default=100, help="LBFGS iterations per refinement step")
    parser.add_argument("--show", help="write a debug PNG with the BODY_25 skeletons drawn on rgb.jpg (single item)")
    parser.add_argument("--coco-out", type=Path, help="also collect all items of the (single) --list into one COCO keypoint "
                        "style json that reuses the images of the SynLoc bbox annotation file of the same split")
    parser.add_argument("--coco-images", type=Path, help="bbox annotation file whose images/annotations are reused "
                        "(default: <coco-out dir>/<split>.json, split = list file name without _vN)")
    parser.add_argument("--threads", type=int, default=default_threads(), help="threads for reading and rewriting scene archives")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be processed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    if args.coco_out is not None:
        if len(args.list) != 1 or args.item_dirs or args.shard:
            parser.error("--coco-out requires exactly one --list and no item directories or --shard")
        coco_images = args.coco_images or default_coco_images(args.coco_out, Path(args.list[0]))
        if not coco_images.exists():
            parser.error(f"bbox annotation file {coco_images} not found; pass --coco-images")
    ARCHIVES.threads = args.threads
    items = iter_items(args)
    if not items:
        parser.error("no items given; pass item directories or --list")
    ARCHIVES.prefetch([item.archive for item in items if item.archive])
    todo, skipped, missing = [], 0, 0
    for item in items:
        absent = [f for f in REQUIRED_FILES if not item.exists(f)]
        if absent:
            missing += 1
            if args.verbose:
                print(f"missing {', '.join(absent)}: {item}", file=sys.stderr)
        elif needs_processing(item, args.overwrite):
            todo.append(item)
        else:
            skipped += 1
    print(f"{len(todo)} items to process, {skipped} already done, {missing} missing or incomplete", file=sys.stderr)
    if args.dry_run:
        return
    if todo:
        convert_items(todo, args)
    if args.coco_out is not None:
        write_coco(args.coco_out, coco_images, Path(args.list[0]), items)


def convert_items(todo: List[Item], args):

    model_paths, vposer_dir = resolve_assets(args.models_dir)
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
    ARCHIVES.flush()

    if stats["humans"]:
        print(f"processed {stats['items']} items, {stats['humans']} humans in {stats['fit_time']:.1f} s "
              f"({stats['humans'] / max(stats['fit_time'], 1e-9):.1f} humans/s); "
              f"joint residual mean {1000 * stats['res_mean_sum'] / stats['humans']:.3f} mm, "
              f"max {1000 * stats['res_max']:.3f} mm; max landmark re-anchoring {1000 * stats['corr_max']:.3f} mm",
              file=sys.stderr)


if __name__ == "__main__":
    main()
