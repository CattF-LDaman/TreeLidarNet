from treelidarnet.metrics import GroundTruth, match_detections
from treelidarnet.postprocess import Detection


def test_match_perfect():
    gts = [GroundTruth("a", 0.0, 0.0)]
    preds = [Detection(0.1, 0.1, 0.9)]
    r = match_detections(preds, gts, match_distance_m=3.0)
    assert r.tp == 1
    assert r.fn == 0
    assert r.f1 == 1.0


def test_match_fp_fn():
    gts = [GroundTruth("a", 0.0, 0.0), GroundTruth("b", 100.0, 100.0)]
    preds = [Detection(50.0, 50.0, 0.8)]
    r = match_detections(preds, gts, match_distance_m=3.0)
    assert r.tp == 0
    assert r.fp == 1
    assert r.fn == 2


def test_audit_excludes_discoveries_from_fp():
    gts = [GroundTruth("a", 0.0, 0.0)]
    preds = [
        Detection(0.1, 0.1, 0.9),
        Detection(500.0, 500.0, 0.8),
    ]
    r = match_detections(preds, gts, match_distance_m=3.0)
    assert r.tp == 1
    assert r.discoveries == 1
    assert r.fp == 1
    assert r.fp_audit == 0
    assert r.precision_audit == 1.0
