(function (global) {
  var STRIDE = 7;
  var HEADER = 12;

  function parseBin(buf) {
    var view = new DataView(buf);
    var magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== "TLEX") throw new Error("bad bundle magic");
    var count = view.getUint32(8, true);
    var f32 = new Float32Array(buf, HEADER, count * STRIDE);
    var lon = new Float32Array(count);
    var lat = new Float32Array(count);
    var bev = new Float32Array(count);
    var v2 = new Float32Array(count);
    var v3 = new Float32Array(count);
    var dist = new Float32Array(count);
    var hasRgb = new Float32Array(count);
    for (var i = 0; i < count; i++) {
      var o = i * STRIDE;
      lon[i] = f32[o];
      lat[i] = f32[o + 1];
      bev[i] = f32[o + 2];
      v2[i] = f32[o + 3];
      v3[i] = f32[o + 4];
      dist[i] = f32[o + 5];
      hasRgb[i] = f32[o + 6];
    }
    return { count: count, lon: lon, lat: lat, bev: bev, v2: v2, v3: v3, dist: dist, hasRgb: hasRgb };
  }

  function parseRegister(buf) {
    var view = new DataView(buf);
    var count = view.getUint32(8, true);
    var f32 = new Float32Array(buf, HEADER, count * 4);
    var out = [];
    for (var i = 0; i < count; i++) {
      var o = i * 4;
      out.push({
        position: [f32[o], f32[o + 1]],
        radiusMeters: f32[o + 2],
        hasSize: f32[o + 3] > 0.5,
      });
    }
    return out;
  }

  async function loadBundle(onStatus) {
    var status = onStatus || function () {};
    status("Reading bundle metadata…");
    var meta = await fetch("/explorer/data/meta.json").then(function (r) {
      if (!r.ok) throw new Error("bundle missing — run make explorer-bundle");
      return r.json();
    });
    status(
      "Downloading " + (meta.detection_count || "?").toLocaleString() + " detections…"
    );
    var detRes = await fetch("/explorer/data/detections.bin");
    if (!detRes.ok) throw new Error("detections.bin missing");
    var detBuf = await detRes.arrayBuffer();
    status("Downloading register (" + (meta.register_count || "?").toLocaleString() + ")…");
    var regRes = await fetch("/explorer/data/register.bin");
    if (!regRes.ok) throw new Error("register.bin missing");
    var regBuf = await regRes.arrayBuffer();
    status("Parsing columnar data…");
    return {
      meta: meta,
      detections: parseBin(detBuf),
      register: parseRegister(regBuf),
    };
  }

  global.ExplorerData = { loadBundle: loadBundle, parseBin: parseBin };
})(window);