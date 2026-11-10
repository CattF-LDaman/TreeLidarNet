(function () {
  "use strict";

  var DATA_URL = "../exports/inference/gent_city_gent_canopy_v2_smoke.json.gz";
  var stats = document.getElementById("stats");

  var map = new maplibregl.Map({
    container: "map",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: [3.7252, 51.0543],
    zoom: 13,
  });

  var overlay = new deck.MapboxOverlay({ interleaved: true, layers: [] });
  map.addControl(overlay);

  var allGt = [];
  var allPred = [];
  var metrics = null;
  var showGt = true;
  var showPred = true;
  var showHeat = false;

  function updateLayers() {
    TreeLidarMap.syncOverlay(overlay, allGt, allPred, {
      showGt: showGt,
      showPred: showPred,
      showPredHeat: showHeat,
    });
    var m = metrics;
    var z = map.getZoom();
    stats.innerHTML =
      "<strong>Smoke:</strong> 200 chips · " +
      allPred.length.toLocaleString() +
      " model · " +
      allGt.length.toLocaleString() +
      " register (citywide GT)<br>z" +
      z.toFixed(1) +
      (m
        ? " · audit F1 <b>" +
          (m.f1_audit_at_3m || m.f1_at_3m).toFixed(3) +
          "</b> · strict " +
          m.f1_at_3m.toFixed(3) +
          " · discoveries " +
          (m.discoveries != null ? m.discoveries : "—")
        : "");
  }

  function fitToPreds() {
    if (!allPred.length) return;
    var lngs = [];
    var lats = [];
    for (var i = 0; i < allPred.length; i++) {
      var p = allPred[i].position;
      lngs.push(p[0]);
      lats.push(p[1]);
    }
    var minLng = Math.min.apply(null, lngs);
    var maxLng = Math.max.apply(null, lngs);
    var minLat = Math.min.apply(null, lats);
    var maxLat = Math.max.apply(null, lats);
    map.fitBounds(
      [
        [minLng, minLat],
        [maxLng, maxLat],
      ],
      { padding: 80, maxZoom: 16, duration: 0 }
    );
  }

  async function loadGzipJson(url) {
    var res = await fetch(url);
    if (!res.ok) throw new Error(url + ": " + res.status);
    var buf = await res.arrayBuffer();
    var ds = new DecompressionStream("gzip");
    var stream = new Blob([buf]).stream().pipeThrough(ds);
    var text = await new Response(stream).text();
    return JSON.parse(text);
  }

  async function boot() {
    try {
      var payload = await loadGzipJson(DATA_URL);
      allGt = payload.gt_trees || [];
      allPred = payload.predictions || [];
      metrics = payload.metrics || null;
      map.once("load", function () {
        fitToPreds();
        updateLayers();
      });
      if (map.loaded()) {
        fitToPreds();
        updateLayers();
      }
    } catch (e) {
      stats.textContent = "Load failed: " + e.message;
    }
  }

  document.getElementById("show-gt").onchange = function (e) {
    showGt = e.target.checked;
    updateLayers();
  };
  document.getElementById("show-pred").onchange = function (e) {
    showPred = e.target.checked;
    updateLayers();
  };
  document.getElementById("show-heat").onchange = function (e) {
    showHeat = e.target.checked;
    updateLayers();
  };

  boot();
})();
