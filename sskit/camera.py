import torch
import numpy as np
import json
from sskit.utils import to_homogeneous, to_cartesian, grid2d, sample_image
from pathlib import Path

def normalize(pkt, image_shape):
    pkt = torch.as_tensor(pkt)
    _, h, w = image_shape
    return (pkt - torch.tensor([(w-1)/2, (h-1)/2], device=pkt.device)) / w

def unnormalize(pkt, image_shape):
    pkt = torch.as_tensor(pkt)
    _, h, w = image_shape
    return w * pkt + torch.tensor([(w-1)/2, (h-1)/2], device=pkt.device)

def world_to_undistorted(camera_matrix, pkt):
    camera_matrix = torch.as_tensor(camera_matrix)
    return to_cartesian(torch.matmul(to_homogeneous(pkt), camera_matrix.mT))

def undistorted_to_ground(camera_matrix, pkt):
    camera_matrix = torch.as_tensor(camera_matrix)
    hom = torch.inverse(camera_matrix[..., [0, 1, 3]])
    pkt = to_cartesian(torch.matmul(to_homogeneous(pkt), hom.mT))
    return torch.cat([pkt, torch.zeros_like(pkt[..., 0:1])], -1)

def _rescale(pkt, rr_in, rr_out, eps=1e-12):
    """pkt scaled from radius rr_in to radius rr_out. The denominator is clamped to eps so that the
    optical axis (rr_in == 0) maps to itself instead of 0 / 0 = nan, and so that a polynomial with
    a nonzero constant term (rr_out(0) != 0) does not turn rounding noise next to the axis into a
    jump of rr_out(0)."""
    return rr_out / rr_in.clamp_min(eps) * pkt

def distort(poly, pkt):
    if isinstance(poly, Spherical):
        return distort_spherical(pkt, poly.fov)
    pkt = torch.as_tensor(pkt)
    poly = torch.as_tensor(poly)
    rr = (pkt ** 2).sum(-1, keepdim=True).sqrt()
    rr2 = polyval(poly, torch.arctan(rr))
    return _rescale(pkt, rr, rr2)

def undistort(poly, pkt):
    if isinstance(poly, Spherical):
        return undistort_spherical(pkt, poly.fov)
    pkt = torch.as_tensor(pkt)
    poly = torch.as_tensor(poly)
    rr2 = (pkt ** 2).sum(-1, keepdim=True).sqrt()
    rr = torch.tan(polyval(poly, rr2))
    return _rescale(pkt, rr2, rr)

SPHERICAL_CLIPPING_ANGLE = 0.499 * np.pi

class Spherical:
    """Spherical lens model, usable in place of a distortion / undistortion polynomial. The image
    x-axis spans the horizontal field of view `fov` radians, i.e. focal length W / fov pixels, with
    the same focal length vertically."""
    def __init__(self, fov=np.pi):
        self.fov = fov

def distort_spherical(pkt, fov=np.pi):
    """Undistorted normalised points (x, y) to normalised image points (pan, tilt) / fov, with
    pan = arctan(x), tilt = arctan(y / sqrt(x^2 + 1)) in radians. Only valid in front of the camera."""
    pkt = torch.as_tensor(pkt)
    x, y = pkt[..., 0], pkt[..., 1]
    return torch.stack([torch.arctan(x), torch.arctan(y / torch.sqrt(x ** 2 + 1))], -1) / fov

def undistort_spherical(pkt, fov=np.pi):
    """Inverse of distort_spherical. The angles are clipped to +-SPHERICAL_CLIPPING_ANGLE to stay
    away from the singularity at +-pi/2."""
    pkt = (torch.as_tensor(pkt) * fov).clamp(-SPHERICAL_CLIPPING_ANGLE, SPHERICAL_CLIPPING_ANGLE)
    x, y = pkt[..., 0], pkt[..., 1]
    return torch.stack([torch.tan(x), torch.tan(y) / torch.cos(x)], -1)

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
    with open(directory / "lens.json") as fd:
        lens = json.load(fd)
    return make_camera(np.load(directory / "camera_matrix.npy"), lens, poly_dim)

def make_camera(camera_matrix, lens: dict, poly_dim=8):
    """Same as load_camera, but from the already loaded contents of camera_matrix.npy and lens.json."""
    camera_matrix = np.asarray(camera_matrix)[:3]
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

def project_on_ground(camera_matrix, dist_poly, image, width=70, height=120, resolution=10, center=(0,0), z=0, padding_mode: str = "zeros"):
    gw, gh = width * resolution, height * resolution
    # pixel centres: pixel ((gw-1)/2, (gh-1)/2) of the ground image maps to `center`
    center = torch.as_tensor(center, device=image.device) - torch.tensor([(gw-1)/2, (gh-1)/2], device=image.device) / resolution
    gnd = grid2d(gw, gh).to(image.device) / resolution + center
    pkt = gnd.reshape(-1, 2)
    pkt = torch.cat([pkt, z * torch.ones_like(pkt[..., 0:1])], -1)
    grid = world_to_image(camera_matrix, dist_poly, pkt).reshape(gnd.shape)
    return sample_image(image, grid[None], padding_mode=padding_mode)

def undistort_image(dist_poly, image, zoom: float = 1.0, padding_mode: str = "zeros", output_size=None,
                    return_k: bool = False, undist_poly=None):
    """Pinhole view of `image` (1, C, H, W) with focal length f = zoom * W pixels: output pixel (u, v)
    looks along the undistorted normalised direction ((u, v) - c) / f, c the centre of the output.
    output_size: None for the input size (which crops the corners of a wide-angle image at zoom < 1),
    (width, height) in pixels, or 'full' for the smallest centred view holding the whole input (the
    input border is traced through `undist_poly`, which is then required). With return_k=True the
    intrinsic matrix [[f, 0, cx], [0, f, cy], [0, 0, 1]] of the view is returned as well."""
    h, w = image.shape[-2:]
    f = w * zoom
    if output_size is None:
        ow, oh = w, h
    elif isinstance(output_size, str) and output_size == "full":
        if undist_poly is None:
            raise ValueError("output_size='full' needs undist_poly")
        xs, ys = torch.arange(w, dtype=torch.float64), torch.arange(h, dtype=torch.float64)
        border = torch.cat([torch.stack([xs, torch.zeros_like(xs)], -1), torch.stack([xs, torch.full_like(xs, h - 1)], -1),
                            torch.stack([torch.zeros_like(ys), ys], -1), torch.stack([torch.full_like(ys, w - 1), ys], -1)])
        ex, ey = (undistort(undist_poly, normalize(border, (1, h, w))) * f).abs().amax(0).ceil().long().tolist()
        ow, oh = 2 * ex + 1, 2 * ey + 1
    else:
        ow, oh = map(int, output_size)   # numpy ints would make the grid float64
    center = torch.tensor([(ow - 1) / 2, (oh - 1) / 2])
    grid = (grid2d(ow, oh) - center).to(image.device) / f
    dgrid = distort(dist_poly, grid).to(image.dtype)
    out = sample_image(image, dgrid[None], padding_mode=padding_mode)
    if not return_k:
        return out
    K = torch.tensor([[f, 0, center[0]], [0, f, center[1]], [0, 0, 1]], dtype=torch.get_default_dtype() if isinstance(dist_poly, Spherical) else torch.as_tensor(dist_poly).dtype, device=image.device)
    return out, K

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

def camera_position(camera_matrix):
    camera_matrix = torch.as_tensor(camera_matrix)
    return -torch.inverse(camera_matrix[..., :3]) @ camera_matrix[..., 3:4]

def look_at(camera_matrix, dist_poly, image, center, zoom=1):
    center = torch.as_tensor(center)
    focal_point = camera_position(camera_matrix)[..., 0]
    pan, tilt = get_pan_tilt_from_direction(center - focal_point)

    rot = camera_matrix[:,:3] @ make_rotation_matrix_from_pan_tilt(pan, tilt).mT
    h, w = image.shape[-2:]
    grid = (grid2d(w, h) - torch.tensor([(w-1)/2, (h-1)/2])) / w / zoom
    rgrid = to_cartesian(torch.matmul(to_homogeneous(grid), rot.mT))
    dgrid = distort(dist_poly, rgrid)
    return sample_image(image, dgrid[None])
