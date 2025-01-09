from pathlib import Path
from sskit import load_camera, imread, world_to_image, Draw, image_to_ground
import json
import torch

d = Path("example")
camera_matrix, dist_poly, undist_poly = load_camera(d)
name = 'pelvis'

img = imread(d / "rgb.jpg")
_, h, w = img.shape
objects = json.loads((d / "objects.json").read_bytes())
pkt = torch.tensor([obj['keypoints'].get(name) for obj in objects.values() if obj['class'] == 'human'])
pkt_img = torch.tensor([obj['keypoints'].get(name + '_img') for obj in objects.values() if obj['class'] == 'human'])

npkt = world_to_image(camera_matrix, dist_poly, pkt)
ipkt = npkt * w + torch.tensor([(w-1)/2, (h-1)/2])
print(((ipkt - pkt_img)**2).sum(1).sqrt().max())
drw = Draw(img)
drw.circle(ipkt, 3, (255,0,0))

# p = world_to_image(camera_matrix, dist_poly, torch.tensor([7., -7., 0.]))
# image_to_ground(camera_matrix, undist_poly, p)

pkt[:,2] = 0
npkt_gnd = world_to_image(camera_matrix, dist_poly, pkt)
ipkt_gnd = npkt_gnd * w + torch.tensor([(w-1)/2), (h-1)/2])
drw.circle(ipkt_gnd, 3, (0,255,0))
drw.line([ipkt, ipkt_gnd], (0,0,255), 2)

drw.save('t.png')
