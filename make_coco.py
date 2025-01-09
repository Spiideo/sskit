from pathlib import Path
from sskit import imshape, world_to_image, load_camera
import json
from tqdm import tqdm
import sys
import numpy as np
import gzip
from io import BytesIO

# d = Path('/home/hakan/data/SoccerCrowdV1')
# out = Path('/home/hakan/data/SpiideoScenes/SoccerCrowd/v1')
d = Path(sys.argv[1])
out = Path(sys.argv[2])

def mkcoco(part):
    imdir = out / part
    imdir.mkdir(parents=True)
    annotations = []
    images = []
    for iid, fn in enumerate(tqdm((d / f'{part}_v1.txt').open().readlines())):
        fn = d / fn.strip()
        ofn = imdir / f'{iid:06d}.jpg'
        ofn.hardlink_to(fn)

        c, h, w = imshape(fn)
        with open(fn.parent / "objects.json") as fd:
            objects = json.load(fd)
        camera_matrix, dist_poly, undist_poly = load_camera(fn.parent)
        afn = fn.parent / "areas_cache.json"
        if afn.exists():
            with afn.open() as fd:
                areas = json.load(fd)
            segs = None
        else:
            segs = np.load(BytesIO(gzip.decompress((fn.parent / "segmentations.npy.gz").read_bytes())))
            areas = {}
        areas_updated = False

        images.append(dict(
            id=iid,
            file_name=ofn.name,
            width=w,
            height=h,
            camera_matrix=camera_matrix.tolist(),
            dist_poly=dist_poly.tolist(),
            undist_poly=undist_poly.tolist(),
        ))


        for oid, obj in sorted(objects.items()):
            if obj['class'] == 'human':
                for key in ["bounding_box_tighter", "bounding_box_tight"]:
                    if key in obj:
                        bbox = obj[key]
                        break
            else:
                continue
            u0, u1, v0, v1 = bbox
            box_w = u1 - u0
            box_h = v1 - v0
            px, py, pz = obj['keypoints']['pelvis']
            pu, pv = (world_to_image(camera_matrix, dist_poly, [px, py, pz]).numpy() * w + ((w-1)/2, (h-1)/2))
            pu2, pv2 = (world_to_image(camera_matrix, dist_poly, [px, py, 0]).numpy() * w + ((w-1)/2, (h-1)/2))

            if oid not in areas:
                areas[oid] = int((segs==obj['segmentation_id']).sum())
                areas_updated = True


            annotations.append(dict(
                id=len(annotations),
                keypoints=[[pu, pv, 1], [pu2, pv2, 1]],
                keypoints_3d=[[px, py, pz, 1], [px, py, 0, 1]],
                position_on_pitch=[px, py],
                bbox=[u0, v0, box_w, box_h],
                area=areas[oid],
                image_id=iid,
                category_id=1,
            ))


        if areas_updated:
            with afn.open('w') as fd:
                json.dump(areas, fd)


    (out / "annotations").mkdir(exist_ok=True)
    with open(out / "annotations" / f"{part}.json", "w") as fd:
        json.dump(dict(images=images, annotations=annotations, categories=[dict(id=1, name="person")]), fd)

mkcoco("mini")
mkcoco("val")
mkcoco("test")
mkcoco("challange")
mkcoco("train")
