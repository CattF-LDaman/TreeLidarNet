from treelidarnet.data.crs import lambert_to_wgs84, wgs84_to_lambert


def test_roundtrip_ghent():
    lon, lat = 3.725, 51.054
    x, y = wgs84_to_lambert(lon, lat)
    lon2, lat2 = lambert_to_wgs84(x, y)
    assert abs(lon2 - lon) < 1e-5
    assert abs(lat2 - lat) < 1e-5
