from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import torch
import PIL.Image
import PIL.ImageDraw
from torch.nn.functional import grid_sample


def imread(fn):
    return read_image(fn).to(torch.get_default_dtype()) / 255

def imwrite(img, fn, rescale=False):
    if rescale:
        img = img - img.min()
        img /= img.max()
    save_image(img, fn)

def immark(im, pos, color=(1,0,0), r=3):
    _, h, w = im.shape
    pos = torch.atleast_2d(pos)
    assert pos.shape[-1] == 2
    for u,v in pos.reshape(-1, 2):
        u, v = int(torch.round(u)), int(torch.round(v))
        if 0 <= u < w and 0 <= v < h:
            im[:, max(v-r, 0):min(v+r, h-1), max(u-r, 0):min(u+r,w-1)] = torch.tensor(color)[:, None, None]
    return im

def imshape(fn):
    im = PIL.Image.open(fn)
    w, h = im.size
    if im.mode != 'RGB':
        raise NotImplementedError
    return (3, h, w)

def to_homogeneous(pkt):
    pkt = torch.as_tensor(pkt)
    return torch.cat([pkt, torch.ones_like(pkt[..., 0:1])], -1)

def to_cartesian(pkt):
    pkt = torch.as_tensor(pkt)
    return pkt[..., :-1] / pkt[..., -1:]

class Draw:
    def __init__(self, img):
        self.pil_img = to_pil_image(img)
        self.draw = PIL.ImageDraw.Draw(self.pil_img)

    def _point_list(self, xy, n=2):
        xy = torch.atleast_2d(torch.as_tensor(xy))
        assert xy.shape[-1] == n
        for pkt in xy.reshape(-1, n):
            yield tuple(map(float, pkt))

    def circle(self, xy, radius, fill=None, outline=None, width=1):
        for pkt in self._point_list(xy):
            self.draw.circle(pkt, radius, fill, outline, width)
        return self

    def line(self, xy, fill=None, width=0, joint=None):
        points = [self._point_list(l) for l in xy]
        for pkts in zip(*points):
            self.draw.line(list(pkts), fill, width, joint)
        return self

    def rectangle(self, xy, fill=None, outline=None, width=1):
        for pkt in self._point_list(xy, 4):
            self.draw.rectangle(pkt, fill, outline, width)
        return self

    def text(self, xy, text, font_size=60):
        for pkt in self._point_list(xy):
            self.draw.text(pkt, text, font_size=font_size)
        return self

    def save(self, fn):
        self.pil_img.save(fn)
        return self

def grid2d(w: int, h: int):
    grid_y = torch.linspace(0.0, h-1, h)
    grid_y = torch.reshape(grid_y, [h, 1])
    grid_y = grid_y.repeat(1, w)

    grid_x = torch.linspace(0.0, w-1, w)
    grid_x = torch.reshape(grid_x, [1, w])
    grid_x = grid_x.repeat(h, 1)

    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid

def sample_image(image, grid):
    n, c, h, w = image.shape
    scaled_grid = 2 * w * grid
    scaled_grid[..., 0] /= (w - 1)
    scaled_grid[..., 1] /= (h - 1)
    return grid_sample(image, scaled_grid, align_corners=True)
