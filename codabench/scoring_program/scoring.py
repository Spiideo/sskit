import json
from pathlib import Path
from xtcocotools.coco import COCO
from sskit.coco import LocSimCOCOeval


reference_dir = Path('/app/input/ref')
prediction_dir = Path('/app/input/res')
score_dir = Path('/app/output')

with open(prediction_dir / 'metadata.json') as fd:
    metadata = json.load(fd)

def eval(tau, suffix):
    coco = COCO(reference_dir / 'gt.json')
    coco_det = coco.loadRes(str(prediction_dir / "results.json"))
    coco_eval = LocSimCOCOeval(coco, coco_det, 'bbox')

    coco_eval.params.useSegm = None
    coco_eval.params.score_threshold = metadata['score_threshold']
    if 'position_from_keypoint_index' in metadata:
        coco_eval.params.position_from_keypoint_index = metadata['position_from_keypoint_index']
    coco_eval.locsim_tau = tau

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_locsim = coco_eval.stats[0]
    precision, recall, f1, score_threshold, frame_accuracy = coco_eval.stats[12:]
    return {
        'mAP-LocSim' + suffix: map_locsim,
        'Precision' + suffix: precision,
        'Recall' + suffix: recall,
        'F1' + suffix: f1,
        'FrameAcc' + suffix: frame_accuracy,
    }

scores = eval(1, '')
scores.update(eval(5, '(t=5)'))

with open(score_dir / 'scores.json', 'w') as fd:
    json.dump(scores, fd)
