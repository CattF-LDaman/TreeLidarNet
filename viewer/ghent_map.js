(function () {
  "use strict";

  var DATA_URL = "../exports/inference/gent_city_gent_real_v1.json.gz";
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
      "z" +
      z.toFixed(1) +
      " · " +
      allGt.length.toLocaleString() +
      " GT · " +
      allPred.length.toLocaleString() +
      " pred" +
      (m
        ? "<br>F1 <b>" +
          m.f1_at_3m.toFixed(2) +
          "</b> audit <b>" +
          (m.f1_audit_at_3m || m.f1_at_3m).toFixed(2) +
          "</b>"
        : "");
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
      map.once("load", updateLayers);
      if (map.loaded()) updateLayers();
    } catch (e) {
      stats.textContent =
        "Load failed: " + e.message + " — run infer-gent or use ghent_live.html";
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
