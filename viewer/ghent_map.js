(function () {
  "use strict";

  var DATA_URL = "../exports/inference/gent_city_gent_canopy_v2.json.gz";
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
  var allPredRaw = [];
  var metrics = null;
  var showGt = true;
  var showPred = true;
  var showHeat = false;
  var scoreMin = 0.35;
  var exportFloor = 0.2;

  function filteredPreds() {
    return allPredRaw.filter(function (d) {
      return (d.score != null ? d.score : 0) >= scoreMin;
    });
  }

  function updateLayers() {
    var preds = filteredPreds();
    TreeLidarMap.syncOverlay(overlay, allGt, preds, {
      showGt: showGt,
      showPred: showPred,
      showPredHeat: showHeat,
    });
    var m = metrics;
    var z = map.getZoom();
    stats.innerHTML =
      "z" +
      z.toFixed(1) +
      " · <b>" +
      preds.length.toLocaleString() +
      "</b> model (confidence ≥ " +
      scoreMin.toFixed(2) +
      ") · " +
      allPredRaw.length.toLocaleString() +
      " exported (floor " +
      exportFloor.toFixed(2) +
      ") · " +
      allGt.length.toLocaleString() +
      " register" +
      (m
        ? "<br>audit F1 <b>" +
          (m.f1_audit_at_3m || m.f1_at_3m).toFixed(2) +
          "</b> · strict " +
          m.f1_at_3m.toFixed(2) +
          " <span style='color:#6e7681'>(at infer export floor)</span>"
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
      allPredRaw = payload.predictions || [];
      metrics = payload.metrics || null;
      exportFloor =
        payload.peak_extract_min_score != null ? payload.peak_extract_min_score : 0.2;
      var threshInput = document.getElementById("score-thresh");
      var threshVal = document.getElementById("thresh-val");
      var threshHelp = document.getElementById("thresh-help");
      if (threshInput) {
        threshInput.min = String(Math.max(0.15, exportFloor));
        threshInput.max = "0.95";
        if (parseFloat(threshInput.value, 10) < exportFloor) {
          threshInput.value = String(Math.max(exportFloor, 0.35));
        }
        scoreMin = parseFloat(threshInput.value, 10);
        if (threshVal) threshVal.textContent = scoreMin.toFixed(2);
        if (threshHelp && payload.peak_extract_note) {
          threshHelp.textContent = payload.peak_extract_note;
        } else if (threshHelp) {
          threshHelp.textContent =
            "Export floor " +
            exportFloor.toFixed(2) +
            "; slide right to hide weaker rings.";
        }
      }
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

  var threshInput = document.getElementById("score-thresh");
  var threshVal = document.getElementById("thresh-val");
  if (threshInput) {
    threshInput.oninput = function (e) {
      scoreMin = parseFloat(e.target.value, 10);
      if (threshVal) threshVal.textContent = scoreMin.toFixed(2);
      updateLayers();
    };
  }

  map.on("zoom", updateLayers);

  boot();
})();
