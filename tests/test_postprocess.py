import numpy as np

from treelidarnet.postprocess import Detection, extract_peaks_from_heatmap, nms_detections


def test_nms_keeps_higher_score():
    dets = [
        Detection(0, 0, 0.9),
        Detection(0.5, 0.5, 0.5),
    ]
    kept = nms_detections(dets, radius_m=2.0)
    assert len(kept) == 1
    assert kept[0].score == 0.9


def test_extract_peaks():
    heat = np.zeros((20, 20), dtype=np.float32)
    heat[10, 10] = 0.95
    peaks = extract_peaks_from_heatmap(heat, origin_x=0.0, origin_y=0.0, score_threshold=0.3)
    assert len(peaks) >= 1
