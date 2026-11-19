from treelidarnet.explorer.app import create_app


def test_explorer_flags_api(tmp_path, monkeypatch):
    flags_path = tmp_path / "explorer_qa_flags.parquet"
    monkeypatch.setattr(
        "treelidarnet.explorer.app.default_flags_path",
        lambda: flags_path,
    )
    app = create_app(bundle_dir=tmp_path / "empty")
    client = app.test_client()

    post = client.post(
        "/explorer/api/flags",
        json={"lon": 3.721, "lat": 51.054, "bev": 0.9, "v2": 0.1, "v3": 0.2},
    )
    assert post.status_code == 200
    body = post.get_json()
    assert body["flag_id"].startswith("fp_")

    listing = client.get("/explorer/api/flags")
    assert listing.status_code == 200
    assert listing.get_json()["count"] == 1

    deleted = client.delete(f"/explorer/api/flags/{body['flag_id']}")
    assert deleted.status_code == 200
    assert client.get("/explorer/api/flags").get_json()["count"] == 0


def test_explorer_route_missing_bundle(tmp_path):
    app = create_app(bundle_dir=tmp_path / "empty")
    client = app.test_client()
    resp = client.get("/explorer")
    assert resp.status_code == 200
    assert b"End-product QA" in resp.data
    meta = client.get("/explorer/data/meta.json")
    assert meta.status_code == 404