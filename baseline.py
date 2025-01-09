import torch
from sskit import imread, Draw, image_to_ground, load_camera, imshape, world_to_image, match
from pathlib import Path
import json

d = Path('example')
fn = d / 'rgb.jpg'
camera_matrix, dist_poly, undist_poly = load_camera(d)

_, h, w = imshape(fn)
objects = json.loads((d / "objects.json").read_bytes())
pkt = torch.tensor([obj['keypoints'].get('pelvis') for obj in objects.values() if obj['class'] == 'human'])
pkt[:, 2] = 0
npkt = world_to_image(camera_matrix, dist_poly, pkt)
ipkt = npkt * w + torch.tensor([(w-1)/2, (h-1)/2])

model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

res = model(fn, 1280)
dets = []
for x1, y1, x2, y2, conf, cls in res.xyxy[0]:
    if res.names[int(cls)] == 'person' and conf > 0.5:
        dets.append([(x1 + x2)/2, y2])
dets = torch.tensor(dets)

bev_dets = image_to_ground(camera_matrix, undist_poly, (dets - torch.tensor([(w-1)/2, (h-1)/2])) / w)

detected, missed, extra, distances, matches = match(pkt, bev_dets)

print(f'Detected {detected}/{detected + missed} players with {extra} false positives with an average detection error of {distances.mean()}')

drw = Draw(imread(fn))
drw.circle(dets, 5, 'red')
drw.circle(ipkt, 5, 'green')
for j, i in matches:
    drw.line([ipkt[i], dets[j]], 'blue', 3)
drw.save('t.png')



