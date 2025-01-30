import torch
from sskit import load_camera, imread, distort, imwrite
from sskit.utils import to_homogeneous, to_cartesian, grid2d, sample_image
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np

pan = -20
tilt = 15
zoom = 1.5

d = Path("example")
camera_matrix, dist_poly, undist_poly = load_camera(d)
img = imread(d / "rgb.jpg")
_, h, w = img.shape

R = Rotation.from_euler('xy', [tilt, pan], degrees=True).as_matrix().astype(np.float32)

grid = (grid2d(w, h) - torch.tensor([(w-1)/2, (h-1)/2])) / w
trfm_grid = to_cartesian(to_homogeneous(grid) @ R.T) / zoom
distorted_grid = distort(dist_poly, trfm_grid)
undistorted_img = sample_image(img[None], distorted_grid[None])

imwrite(undistorted_img, "t.jpg")
