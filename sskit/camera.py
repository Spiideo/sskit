import torch
import numpy as np
import json
from sskit.utils import to_homogeneous, to_cartesian, grid2d, sample_image
from pathlib import Path

def normalize(pkt, image_shape):
    pkt = torch.as_tensor(pkt)
    _, h, w = image_shape
    return (pkt - torch.tensor([w/2, h/2])) / w

def unnormalize(pkt, image_shape):
    pkt = torch.as_tensor(pkt)
    _, h, w = image_shape
    return w * pkt + torch.tensor([w/2, h/2])

def world_to_undistorted(camera_matrix, pkt):
    camera_matrix = torch.as_tensor(camera_matrix)
    return to_cartesian(torch.matmul(to_homogeneous(pkt), camera_matrix.mT))

def undistorted_to_ground(camera_matrix, pkt):
    camera_matrix = torch.as_tensor(camera_matrix)
    hom = torch.inverse(camera_matrix[..., [0, 1, 3]])
    pkt = to_cartesian(torch.matmul(to_homogeneous(pkt), hom.mT))
    return torch.cat([pkt, torch.zeros_like(pkt[..., 0:1])], -1)

def distort(poly, pkt):
    pkt = torch.as_tensor(pkt)
    poly = torch.as_tensor(poly)
    rr = (pkt ** 2).sum(-1, keepdim=True).sqrt()
    rr2 = polyval(poly, torch.arctan(rr))
    scale = rr2 / rr
    return scale * pkt

def undistort(poly, pkt):
    pkt = torch.as_tensor(pkt)
    poly = torch.as_tensor(poly)
    rr2 = (pkt ** 2).sum(-1, keepdim=True).sqrt()
    rr = torch.tan(polyval(poly, rr2))
    scale = rr / rr2
    return scale * pkt


def polyval(poly, pkt):
    sa = poly[..., 0:1]
    for i in range(1, poly.shape[-1]):
        sa = pkt * sa + poly[..., i]
    return sa

def world_to_image(camera_matrix, distortion_poly, pkt):
    return distort(distortion_poly, world_to_undistorted(camera_matrix, pkt))

def image_to_ground(camera_matrix, undistortion_poly, pkt):
    return undistorted_to_ground(camera_matrix, undistort(undistortion_poly, pkt))

def load_camera(directory: Path, poly_dim=8):
    directory = Path(directory)
    camera_matrix = np.load(directory / "camera_matrix.npy")[:3]
    with open(directory / "lens.json") as fd:
        lens = json.load(fd)
    dist_poly = lens["dist_poly"]
    sensor_width = lens["sensor_width"]
    pixel_width = lens["pixel_width"]

    def d2a(dist_poly, x):
        return -sum(k * x ** i for i, k in enumerate(dist_poly)) / 180 * np.pi

    rr2 = np.linspace(0, 1.5, 200)
    rr = d2a(dist_poly, rr2 * sensor_width * pixel_width)
    msk = (0 <= rr) & (rr < 1.5)
    rr2 = rr2[msk]
    rr = rr[msk]
    poly = np.polyfit(rr, rr2, poly_dim)
    rev_poly = np.polyfit(rr2, rr, poly_dim)

    t = torch.get_default_dtype()
    camera_matrix_t = torch.tensor(camera_matrix).to(t)
    poly_t = torch.tensor(poly).to(t)
    rev_poly_t = torch.tensor(rev_poly).to(t)
    return camera_matrix_t, poly_t, rev_poly_t

def project_on_ground(camera_matrix, dist_poly, image, width=70, height=120, resolution=10, center=(0,0), z=0):
    center = torch.as_tensor(center, device=image.device) - torch.tensor([width/2, height/2], device=image.device)
    gnd = grid2d(width * resolution, height * resolution).to(image.device) / resolution + center
    pkt = gnd.reshape(-1, 2)
    pkt = torch.cat([pkt, z * torch.ones_like(pkt[..., 0:1])], -1)
    grid = world_to_image(camera_matrix, dist_poly, pkt).reshape(gnd.shape)
    return sample_image(image, grid[None])

def undistort_image(dist_poly, image, zoom=1):
    h, w = image.shape[-2:]
    grid = (grid2d(w, h) - torch.tensor([w/2, h/2])) / w / zoom
    dgrid = distort(dist_poly, grid)
    return sample_image(image, dgrid[None])

def get_pan_tilt_from_direction(direction):
    direction = torch.as_tensor(direction)
    x, y, z = direction
    return torch.arctan2(-y, x), torch.arctan2(z, torch.sqrt(x**2 + y**2))

def make_rotation_matrix_from_pan_tilt(pan: float, tilt: float):
    pan = torch.as_tensor(pan)
    tilt = torch.as_tensor(tilt)
    cp = torch.cos(pan)
    sp = torch.sin(pan)
    ct = torch.cos(tilt)
    st = torch.sin(tilt)
    return torch.tensor(((-sp, -cp, 0), (st * cp, -st * sp, -ct), (ct * cp, -ct * sp, st)))

def look_at(camera_matrix, dist_poly, image, center, zoom=1):
    center = torch.as_tensor(center)
    focal_point = -torch.inverse(camera_matrix[:,:3]) @ camera_matrix[:, 3]
    pan, tilt = get_pan_tilt_from_direction(center - focal_point)

    rot = camera_matrix[:,:3] @ make_rotation_matrix_from_pan_tilt(pan, tilt).mT
    h, w = image.shape[-2:]
    grid = (grid2d(w, h) - torch.tensor([w/2, h/2])) / w / zoom
    rgrid = to_cartesian(torch.matmul(to_homogeneous(grid), rot.mT))
    dgrid = distort(dist_poly, rgrid)
    return sample_image(image, dgrid[None])
