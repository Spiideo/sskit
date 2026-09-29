from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import contextlib, io
import numpy as np
import torch
from sskit import image_to_ground, world_to_image, unnormalize
from sskit.pose import BODY25_NAMES, BODY25_SIGMAS, BODY25_TO_FIFA15, BODY25_TO_COCO17

def locsim(dist2, tau=1):
    """LocSim of squared distances: 1 at distance 0, 0.05 at distance tau."""
    return np.exp(np.log(0.05) * dist2 / tau**2)

class LocSimCOCOeval(COCOeval):
    locsim_tau = 1

    def get_img_pos(self, dt):
        return [np.array(det['keypoints']).reshape(-1,3)[self.params.position_from_keypoint_index, :2] for det in dt]

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 or len(dt) == 0:
            return []
        inds = np.argsort([-d[self.score_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        img = self.cocoGt.loadImgs(int(imgId))[0]
        if hasattr(self.params, 'position_from_keypoint_index'):
            img_pos_dt = np.array(self.get_img_pos(dt))
            w, h = np.float32(img['width']), np.float32(img['height'])
            nimg_pos_dt = ((img_pos_dt - ((w-1)/2, (h-1)/2)) / w).astype(np.float32)
            bev_dt = image_to_ground(img['camera_matrix'], img['undist_poly'], nimg_pos_dt)[:, :2]
        else:
            bev_dt = np.array([det['position_on_pitch'] for det in dt])
        bev_gt = np.array([det['position_on_pitch'] for det in gt])

        aa, bb = np.meshgrid(bev_gt[:,0], bev_dt[:,0])
        dist2 = (aa - bb) ** 2
        aa, bb = np.meshgrid(bev_gt[:,1], bev_dt[:,1])
        dist2 += (aa - bb) ** 2

        return locsim(dist2, self.locsim_tau)

    def accumulate(self, p=None):
        if p is None:
            p = self.params
        super().accumulate(p)

        iou = p.iouThrs == 0.5
        area = p.areaRngLbl.index('all')
        dets = np.argmax(p.maxDets)

        precision = np.squeeze(self.eval['precision'][iou, :, 0, area, dets])
        scores = np.squeeze(self.eval['scores'][iou, :, 0, area, dets])
        recall = p.recThrs
        f1 = 2 * precision * recall / (precision + recall)

        self.eval['precision_50'] = precision
        self.eval['recall_50'] = recall
        self.eval['f1_50'] = f1
        self.eval['scores_50'] = scores

    def frame_accuracy(self, threshold):
        rng = self.params.areaRng[self.params.areaRngLbl.index('all')]
        iou = self.params.iouThrs == 0.5

        ok = bad = 0
        for e in self.evalImgs:
            if e is None:
                continue
            if e['aRng'] == rng:
                matches = (e['dtMatches'][iou] > -1)[0]
                if (np.array(e['dtScores'])[matches] > threshold).sum() == len(e['gtIds']):
                    ok += 1
                else:
                    bad += 1
        return ok / (ok + bad)

    def summarize(self):
        super().summarize()
        self.summarize_locsim()

    def summarize_locsim(self):
        if hasattr(self.params, 'score_threshold'):
            threshold = self.params.score_threshold
        else:
            i = self.eval['f1_50'].argmax()
            scores = self.eval['scores_50']
            # midway to the next (lower) score; 0 when the best F1 is reached at the last recall bin
            threshold = (scores[i] + (scores[i + 1] if i + 1 < len(scores) else 0)) / 2
        i = np.searchsorted(-self.eval['scores_50'], -threshold, 'right') - 1
        stats = [self.eval['precision_50'][i], self.eval['recall_50'][i], self.eval['f1_50'][i], threshold, self.frame_accuracy(threshold)]
        self.stats = np.concatenate([self.stats, stats])

        print()
        print(f'  Precision      @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[0]:5.3f}')
        print(f'  Recall         @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[1]:5.3f}')
        print(f'  F1             @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[2]:5.3f}')
        print(f'  Frame Accuracy @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[4]:5.3f}')
        print(f'  mAP-LocSim     @[ LocSim=0.50:0.95 | ScoreTh={threshold:5.3f} ] = {self.stats[0]:5.3f}')


class BBoxLocSimCOCOeval(LocSimCOCOeval):
    def get_img_pos(self, dt):
        def bbox_ground(x, y, w, h):
            return (x + w/2, y + h)
        return [bbox_ground(*det['bbox']) for det in dt]


class Keypoint3DLocSimCOCOeval(LocSimCOCOeval):
    """Keypoint evaluation (iouType 'keypoints') that matches detections to ground truth on the
    mean LocSim of the 3D distances between the detected and ground truth `keypoints_3d`, taken
    over the ground truth keypoints with visibility > 0, instead of on the OKS. `keypoints_3d`
    holds one [x, y, z] or [x, y, z, 1] row per keypoint (nested, or flat, in which case rows of
    3 are assumed unless the length is not divisible by 3). Up to `max_dets`
    detections per image are evaluated (xtcocotools' keypoint default of 20 is too low for a
    soccer frame)."""

    def __init__(self, cocoGt=None, cocoDt=None, iouType='keypoints', max_dets=100, **kwargs):
        super().__init__(cocoGt, cocoDt, iouType, **kwargs)
        self.params.maxDets = [max_dets]

    def summarize(self):
        # like xtcocotools' keypoint summary, but for our maxDets instead of the hard-coded 20
        p = self.params
        max_dets = p.maxDets[-1]

        def stat(ap, iou_thr=None, area='all'):
            s = self.eval['precision' if ap else 'recall']
            if iou_thr is not None:
                s = s[np.isclose(p.iouThrs, iou_thr)]
            s = s[..., p.areaRngLbl.index(area), p.maxDets.index(max_dets)]
            v = np.mean(s[s > -1]) if (s > -1).any() else -1
            iou = f'{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}' if iou_thr is None else f'{iou_thr:0.2f}'
            print(f' {"Average Precision" if ap else "Average Recall":<18} {"(AP)" if ap else "(AR)"} '
                  f'@[ IoU={iou:<9} | area={area:>6s} | maxDets={max_dets:>3d} ] = {v: 0.3f}')
            return v

        self.stats = np.array([stat(1), stat(1, .5), stat(1, .75), stat(1, area='medium'), stat(1, area='large'),
                               stat(0), stat(0, .5), stat(0, .75), stat(0, area='medium'), stat(0, area='large')])
        self.summarize_locsim()

    @staticmethod
    def get_kp3d(anns):
        def rows(kp):
            kp = np.asarray(kp, float)
            return kp.reshape(-1, 4 if kp.size % 3 else 3) if kp.ndim == 1 else kp
        return np.array([rows(a['keypoints_3d'])[:, :3] for a in anns])

    def computeOks(self, imgId, catId):
        p = self.params
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d[self.score_key] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []
        d = self.get_kp3d(dts)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            g = self.get_kp3d([gt])[0]
            vis = np.asarray(gt['keypoints'])[2::3] > 0
            if vis.any():
                dist2 = ((d[:, vis] - g[vis]) ** 2).sum(-1)
                ious[:, j] = locsim(dist2, self.locsim_tau).mean(1)
        return ious


def project_keypoints(image, kp3d):
    """COCO keypoints [u, v, 1, ...] in pixels of the 3D keypoints `kp3d` projected into `image`
    (a COCO image entry with `camera_matrix`, `dist_poly`, `width` and `height`)."""
    pkt = torch.as_tensor(Keypoint3DLocSimCOCOeval.get_kp3d([{'keypoints_3d': kp3d}])[0], dtype=torch.float32)
    uv = world_to_image(torch.tensor(image['camera_matrix']), torch.tensor(image['dist_poly']), pkt)
    uv = unnormalize(uv, (3, image['height'], image['width'])).numpy()
    return np.c_[uv, np.ones(len(uv))].ravel().tolist()


LOCSIM_KEYPOINT_SUBSETS = {
    "body25-3d-locsim": list(range(25)),
    "fifa15-3d-locsim": BODY25_TO_FIFA15,
    "coco-3d-locsim": BODY25_TO_COCO17,
}


def coco_eval(gt_path, res, iou_type, log, exclude=(), locsim_tau=1):
    """xtcocotools evaluation; keypoints listed in `exclude` are marked invisible in the GT
    so that they do not enter the OKS. The iou_types "body25-3d-locsim", "fifa15-3d-locsim" and
    "coco-3d-locsim" use Keypoint3DLocSimCOCOeval, i.e. the mean LocSim (with tau `locsim_tau`
    metres) of the 3D keypoint distances over the visible ground truth joints of the BODY25,
    FIFA15 or COCO17 subset, in place of the OKS. Detections then need `keypoints_3d`; their 2D
    `keypoints` are optional and projected from `keypoints_3d` when missing (they only provide
    the detection area used by the area ranges)."""
    subset = LOCSIM_KEYPOINT_SUBSETS.get(iou_type)
    hidden = set(exclude)
    if subset is not None:
        hidden |= set(range(25)) - set(subset)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO(gt_path)
        if (iou_type == "keypoints" or subset is not None) and hidden:
            for a in coco_gt.dataset["annotations"]:
                for j in hidden:
                    if a["keypoints"][3 * j + 2] > 0:
                        a["keypoints"][3 * j + 2] = 0
                        a["num_keypoints"] -= 1
        if subset is not None:
            res = [d if 'keypoints' in d else dict(d, keypoints=project_keypoints(coco_gt.imgs[d['image_id']], d['keypoints_3d']))
                   for d in res]
        coco_dt = coco_gt.loadRes(res) if res else COCO()
        if subset is not None:
            ev = Keypoint3DLocSimCOCOeval(coco_gt, coco_dt, "keypoints")
            ev.locsim_tau = locsim_tau
        else:
            # xtcocotools takes the OKS sigmas in the constructor (params.kpt_oks_sigmas is ignored)
            ev = COCOeval(coco_gt, coco_dt, iou_type,
                          sigmas=BODY25_SIGMAS if iou_type == "keypoints" else None)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    text = buf.getvalue()
    text = text[text.find(" Average Precision"):] if " Average Precision" in text else text
    note = f", without {', '.join(BODY25_NAMES[j] for j in exclude)}" if exclude and iou_type != "bbox" else ""
    log(f"== COCO {iou_type} ({len(res)} detections{note})\n{text.rstrip()}")
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"] \
        if iou_type == "bbox" else ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
    if subset is not None:
        names += ["precision_50", "recall_50", "f1_50", "score_threshold", "frame_accuracy"]
    return dict(zip(names, [float(s) for s in ev.stats]))


