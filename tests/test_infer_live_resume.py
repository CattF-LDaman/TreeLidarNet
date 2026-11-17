import orjson

from treelidarnet.live_export import read_live_predictions, write_live_predictions
from treelidarnet.postprocess import Detection, merge_detection_lists
def test_read_live_predictions_roundtrip(tmp_path):
    dets = [
        Detection(x=103_517.2, y=195_842.1, score=0.91),
        Detection(x=103_520.0, y=195_850.0, score=0.88),
    ]
    path = tmp_path / "predictions.json"
    write_live_predictions(path, dets)
    loaded = read_live_predictions(path)
    assert len(loaded) == 2
    assert abs(loaded[0].x - dets[0].x) < 0.05
    assert abs(loaded[0].y - dets[0].y) < 0.05


def test_write_live_predictions_dual_scores(tmp_path):
    dets = [Detection(x=103_517.2, y=195_842.1, score=0.91, verifier_score=0.6)]
    path = tmp_path / "predictions.json"
    write_live_predictions(
        path,
        dets,
        verifier_scores_list=[{"v2": 0.42, "v3": 0.61}],
    )
    payload = orjson.loads(path.read_bytes())
    assert payload["compare"] is True
    row = payload["predictions"][0]
    assert row["verifier_scores"] == {"v2": 0.42, "v3": 0.61}
    assert row["verifier_score"] == 0.61


def test_merge_detection_lists_dedupes():
    a = [Detection(x=0.0, y=0.0, score=0.9)]
    b = [Detection(x=0.1, y=0.0, score=0.85)]
    merged = merge_detection_lists(a, b, radius_m=2.5)
    assert len(merged) == 1
    assert merged[0].score == 0.9