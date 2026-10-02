"""distort / undistort on and next to the optical axis (no example data needed)."""
import numpy as np
import torch

from sskit import distort, undistort

# SpiideoSynLoc mini image 0: both polynomials have a nonzero constant term
DIST_POLY = [0.03668929636478424, -0.18116715550422668, 0.26817721128463745, 0.03681742399930954,
             -0.40782368183135986, 0.25091075897216797, -0.11876903474330902, 0.7688610553741455,
             -0.0007075478788465261]
UNDIST_POLY = [3.73205502959828e-12, -1.1132051010165345e-11, 1.271311094591665e-11, -6.655078831768746e-12,
               1.0398279428482056, -0.5045416951179504, 0.23418931663036346, 1.3027212619781494,
               0.0008982417639344931]


def reference(poly, pkt, fn):
    """The unguarded formula: scale by r_out / r_in."""
    pkt = np.asarray(pkt, np.float64)
    r = np.linalg.norm(pkt, axis=-1, keepdims=True)
    poly = np.asarray(poly, np.float64)
    r_out = np.tan(np.polyval(poly, r)) if fn is undistort else np.polyval(poly, np.arctan(r))
    return pkt * r_out / r


def test_axis_is_fixed_point():
    for fn, poly in ((distort, DIST_POLY), (undistort, UNDIST_POLY)):
        for pkt in (torch.zeros(2, dtype=torch.float64), torch.zeros(3, 4, 2), torch.tensor([[0.0, 0.0], [0.1, -0.2]])):
            out = fn(poly, pkt)
            assert out.shape == pkt.shape
            assert torch.isfinite(out).all()
            assert torch.equal(out[..., :1, :] if pkt.dim() > 1 else out, torch.zeros_like(out[..., :1, :] if pkt.dim() > 1 else out))


def test_no_jump_next_to_axis():
    # a point 1e-15 from the axis stays within 1e-15 * r_out(0) / 1e-12 of it, instead of jumping to r_out(0)
    for fn, poly in ((distort, DIST_POLY), (undistort, UNDIST_POLY)):
        near = torch.tensor([[1e-15, 0.0], [0.0, -3e-17], [-1e-13, 5e-14]], dtype=torch.float64)
        out = fn(poly, near)
        assert torch.isfinite(out).all()
        assert (out.abs() <= abs(poly[-1]) * near.norm(dim=-1, keepdim=True) / 1e-12 * 1.01).all()
        # at 1e-12 and beyond the guard is inactive and the plain formula applies
        far = torch.tensor([[1e-12, 0.0], [1e-6, 2e-6], [0.3, -0.2], [-0.45, 0.1]], dtype=torch.float64)
        np.testing.assert_allclose(fn(poly, far).numpy(), reference(poly, far.numpy(), fn), rtol=1e-12, atol=0)


def test_round_trip():
    pkt = torch.tensor(np.random.default_rng(0).uniform(-0.4, 0.4, (50, 2)))
    np.testing.assert_allclose(distort(DIST_POLY, undistort(UNDIST_POLY, pkt)).numpy(), pkt.numpy(), atol=2e-3)


def test_undistort_image_output_size():
    """Resampling coordinate ramps returns the sampling coordinates: the output pixel (u, v) of a
    width x height view reads the input at unnormalize(distort(((u, v) - c_out) / (w * zoom)))."""
    from sskit import undistort_image, unnormalize, normalize
    from sskit.utils import grid2d
    w, h, zoom = 64, 48, 0.8
    poly = torch.tensor([-0.2, 0.0, 1.0, 0.0])                  # r_d = theta - 0.2 theta^3
    theta = np.linspace(0, 1.0, 200)
    undist = torch.tensor(np.polyfit(np.polyval(poly.numpy(), theta), theta, 8), dtype=torch.float32)
    ramps = grid2d(w, h).permute(2, 0, 1)[None]                   # channel 0 = x, 1 = y
    for output_size in (None, (96, 70), (40, 30), (np.int64(96), np.int64(70)), "full"):
        out, K = undistort_image(poly, ramps, zoom, output_size=output_size, return_k=True, undist_poly=undist)
        ow, oh = out.shape[-1], out.shape[-2]
        if output_size is None:
            assert (ow, oh) == (w, h)
        elif output_size != "full":
            assert (ow, oh) == tuple(int(v) for v in output_size)
        assert torch.allclose(K, torch.tensor([[w * zoom, 0, (ow - 1) / 2], [0, w * zoom, (oh - 1) / 2], [0, 0, 1]]))
        n = (grid2d(ow, oh) - torch.tensor([(ow - 1) / 2, (oh - 1) / 2])) / w / zoom
        want = unnormalize(distort(poly, n), (2, h, w)).permute(2, 0, 1)
        inside = (want[0] >= 0) & (want[0] <= w - 1) & (want[1] >= 0) & (want[1] <= h - 1)
        assert torch.allclose(out[0][:, inside], want[:, inside], atol=1e-3)
        outside = (want[0] < -1) | (want[0] > w) | (want[1] < -1) | (want[1] > h)
        assert (out[0][:, outside] == 0).all()
    # 'full': the input corners land inside the view, within a pixel of its border (ceil)
    out, K = undistort_image(poly, ramps, zoom, output_size="full", return_k=True, undist_poly=undist)
    corners = torch.tensor([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=torch.float32)
    q = undistort(undist, normalize(corners, (1, h, w))) * (w * zoom) + K[:2, 2]
    assert (q >= 0).all() and (q[:, 0] <= out.shape[-1] - 1).all() and (q[:, 1] <= out.shape[-2] - 1).all()
    assert q[:, 0].max() > out.shape[-1] - 2 and q[:, 1].max() > out.shape[-2] - 2
    assert out.shape[-1] > w and out.shape[-2] > h                # wide angle at zoom < 1: bigger than the input
    # the default is unchanged, and without return_k only the image comes back
    assert torch.equal(undistort_image(poly, ramps, zoom), undistort_image(poly, ramps, zoom, output_size=(w, h)))
    try:
        undistort_image(poly, ramps, zoom, output_size="full")
        assert False, "needs undist_poly"
    except ValueError:
        pass


def test_spherical_round_trip():
    from sskit import distort_spherical, undistort_spherical
    pkt = torch.tensor([[0.0, 0.0], [0.3, -0.2], [-2.0, 1.5], [5.0, 0.1]], dtype=torch.float64)
    ang = distort_spherical(pkt)
    np.testing.assert_allclose(ang[:, 0].numpy(), np.arctan(pkt[:, 0].numpy()) / np.pi)
    np.testing.assert_allclose(undistort_spherical(ang).numpy(), pkt.numpy(), rtol=1e-12, atol=1e-12)
    assert torch.isfinite(undistort_spherical(torch.tensor([[0.5, 0.5]]))).all()
