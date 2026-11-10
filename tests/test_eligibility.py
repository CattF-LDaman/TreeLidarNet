from treelidarnet.data.eligibility import eligibility, parse_aanlegjaar


def test_aanleg_after_lidar_ineligible():
    ok, reason = eligibility(aanlegjaar="2020", beheerfase="", sortiment="Acer")
    assert not ok
    assert reason == "aanlegjaar_after_lidar"


def test_empty_aanleg_eligible():
    ok, _ = eligibility(aanlegjaar="", beheerfase="", sortiment="Acer")
    assert ok


def test_beheerfase_exclude():
    ok, reason = eligibility(aanlegjaar="", beheerfase="Afbraak gepland", sortiment="")
    assert not ok
    assert reason is not None


def test_parse_year():
    assert parse_aanlegjaar("2010") == 2010
    assert parse_aanlegjaar("") is None
