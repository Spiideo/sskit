from pathlib import Path
from sskit import imread, Draw, image_to_ground, load_camera, imshape, world_to_image, match
import json
import cv2
import torch
from torchvision.ops import box_iou
import numpy as np
from matplotlib import pyplot as plt

d = Path('example')
fn = d / 'rgb.jpg'

camera_matrix, dist_poly, undist_poly = load_camera(d)
_, h, w = imshape(fn)
objects = json.loads((d / "objects.json").read_bytes())
pkt = torch.tensor([obj['keypoints'].get('pelvis') for obj in objects.values() if obj['class'] == 'human'])
pkt[:, 2] = 0
npkt = world_to_image(camera_matrix, dist_poly, pkt)
ipkt = npkt * w + torch.tensor([(w-1)/2, (h-1)/2])

bbox = torch.tensor([obj['bounding_box_tight'] for obj in objects.values() if obj['class'] == 'human'])
dets = torch.column_stack([bbox[:, :2].mean(1, dtype=torch.float32), bbox[:,3]])

def bbox_to_bev(camera_matrix, undist_poly, bbox):
    dets = torch.column_stack([bbox[:, :2].mean(1, dtype=torch.float32), bbox[:,3]])
    return image_to_ground(camera_matrix, undist_poly, (dets - torch.tensor([(w-1)/2, (h-1)/2])) / w)

def jitter_bbox(bbox, amount=0.5):
    bbox = torch.as_tensor(bbox, dtype=torch.float32)
    bbox_center_x = (bbox[:,1] + bbox[:,0]) / 2
    bbox_center_y = (bbox[:,3] + bbox[:,2]) / 2
    bbox_width = bbox[:,1] - bbox[:,0]
    bbox_height = bbox[:,3] - bbox[:,2]

    scalex = 1 + (2 * torch.rand(len(bbox)) - 1) * amount
    scaley = 1 + (2 * torch.rand(len(bbox)) - 1) * amount
    dx = (2 * torch.rand(len(bbox)) - 1) * bbox_width * amount
    dy = (2 * torch.rand(len(bbox)) - 1) * bbox_height * amount

    bbox_center_x += dx
    bbox_center_y += dy
    bbox_width *= scalex
    bbox_height *= scaley

    return torch.column_stack([
        bbox_center_x - bbox_width/2,
        bbox_center_x + bbox_width/2,
        bbox_center_y - bbox_height/2,
        bbox_center_y + bbox_height/2,
    ])


dists = []
ious = []
jittered_bbox = bbox
for _ in range(1000):
    ious.append(torch.diag(box_iou(bbox[:, [0, 2, 1, 3]], jittered_bbox[:, [0, 2, 1, 3]])).numpy())
    bev_dets = bbox_to_bev(camera_matrix, undist_poly, jittered_bbox)
    dists.append(((bev_dets - pkt)**2).sum(1).sqrt().numpy())
    jittered_bbox = jitter_bbox(bbox, 0.25)
dists_perfect = dists[0]
ious_perfext = ious[0]
dists_box0 = np.array([d[0] for d in dists[1:]])
ious_box0 = np.array([d[0] for d in ious[1:]])
dists = np.concatenate([d[1:] for d in dists[1:]])
ious = np.concatenate([d[1:] for d in ious[1:]])

plt.plot(ious, dists, '*', color=colors[0])
plt.plot(ious_perfext, dists_perfect, '*', color=colors[3])
plt.plot(ious_box0, dists_box0, '*', color=colors[2])
plt.xlabel('IoU')
plt.ylabel('BEV Distance (m)')
plt.show()

colors = ['#4a2377', '#f55f74', '#8cc5e3', '#0d7d87']
names = ['purple', 'pink', 'blue', 'teal']

drw = Draw(imread(fn))
drw.rectangle(bbox[:, [0, 2, 1, 3]], outline=colors[0], width=5)
drw.rectangle(bbox[0, [0, 2, 1, 3]], outline=colors[2], width=5)
drw.circle(dets, 5, colors[3])
drw.circle(ipkt, 5, colors[1])
drw.save('t.png')