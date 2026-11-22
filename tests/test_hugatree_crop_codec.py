import numpy as np

from treelidarnet.integrations.hugatree_crop_codec import decode_crop_blob, encode_crop_tensor


def test_crop_pack_round_trip():
    arr = np.array(
        [
            [1.234, -0.5, 12.3, 0.42, 0, 0, 0.8, 0.6, 0.2],
            [-2.0, 3.25, 0.05, 0.1, 0, 0, 0.1, 0.9, 0.3],
        ],
        dtype=np.float32,
    )
    blob = encode_crop_tensor(arr)
    pts = decode_crop_blob(blob)
    assert len(pts) == 2
    assert abs(pts[0]["x"] - 1.234) < 0.002
    assert abs(pts[0]["intensity"] - 1.0) < 0.02
    assert abs(pts[1]["intensity"] - (0.1 / 0.42)) < 0.02
    assert abs(pts[0]["red"] - 0.8) < 0.01