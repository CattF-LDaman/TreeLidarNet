import numpy as np

from treelidarnet.postprocess import Detection, extract_peaks_from_heatmap, nms_detections


def test_merge_matches_score_ordered_scan():
    from treelidarnet.postprocess import merge_chip_detections

    rng = np.random.default_rng(0)
    dets = []
    for i in range(80):
        dets.append(
            (
                f"chip{i % 3}",
                [
                    Detection(
                        float(rng.uniform(0, 40)),
                        float(rng.uniform(0, 40)),
                        float(rng.uniform(0.2, 1)),
                        source_chip_ids=(f"chip{i % 3}",),
                    )
                ],
            )
        )
    kept = merge_chip_detections(dets, radius_m=2.5)
    brute = _brute_merge(dets, 2.5)
    assert [(d.x, d.y, d.score) for d in kept] == [(d.x, d.y, d.score) for d in brute]


def _brute_merge(chip_detections, radius_m):
    all_d = []
    for chip_id, group in chip_detections:
        for d in group:
            all_d.append(
                Detection(
                    x=d.x,
                    y=d.y,
                    score=d.score,
                    source_chip_ids=(chip_id,) + d.source_chip_ids,
                )
            )
    sorted_d = sorted(all_d, key=lambda d: d.score, reverse=True)
    kept = []
    r2 = radius_m * radius_m
    for cand in sorted_d:
        ok = True
        for k in kept:
            dx = cand.x - k.x
            dy = cand.y - k.y
            if dx * dx + dy * dy <= r2:
                ok = False
                break
        if ok:
            kept.append(cand)
    return kept


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
