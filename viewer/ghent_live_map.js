(function () {
  "use strict";

  var LIVE = "../exports/inference/live/";
  var POLL_MS = 2000;
  var stats = document.getElementById("stats");
  var prog = document.getElementById("prog");
  var tip = document.getElementById("tip");

  var map = new maplibregl.Map({
    container: "map",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: [3.7252, 51.0543],
    zoom: 14,
  });

  var overlay = new deck.MapboxOverlay({
    interleaved: true,
    layers: [],
    onHover: function (info) {
      if (info.object && info.layer && info.x != null) {
        var d = info.object;
        var layerId = info.layer.id;
        if (layerId === "ghent-gt-trees") {
          tip.style.display = "block";
          tip.style.left = info.x + 12 + "px";
          tip.style.top = info.y + 12 + "px";
          tip.innerHTML =
            "<strong style='color:#34d399'>Register tree</strong><br>" +
            (d.sortiment || "") +
            (d.gis_id ? "<br><span style='color:#8b949e'>" + d.gis_id + "</span>" : "");
        } else if (layerId === "ghent-pred-rings") {
          tip.style.display = "block";
          tip.style.left = info.x + 12 + "px";
          tip.style.top = info.y + 12 + "px";
          tip.innerHTML =
            "<strong style='color:#fbbf24'>Model prediction</strong><br>score " +
            (d.score != null ? d.score.toFixed(2) : "?");
        }
      } else {
        tip.style.display = "none";
      }
    },
  });
  map.addControl(overlay);

  var allGt = [];
  var allPred = [];
  var lastPredKey = "";
  var pollTimer = null;
  var showGt = true;
  var showPred = true;

  function updateLayers() {
    TreeLidarMap.syncOverlay(overlay, allGt, allPred, {
      showGt: showGt,
      showPred: showPred,
      showPredHeat: false,
    });
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

  async function ensureGt() {
    if (allGt.length) return;
    var data = await loadGzipJson(LIVE + "gt.json.gz");
    allGt = data.gt_trees || [];
    updateLayers();
  }

  async function poll() {
    try {
      await ensureGt();
      var stRes = await fetch(LIVE + "status.json?t=" + Date.now());
      if (!stRes.ok) {
        stats.textContent =
          "Start: uv run treelidarnet infer-gent --live --checkpoint runs/gent_real_v1/best.pt";
        return;
      }
      var status = await stRes.json();
      prog.style.width = status.total ? (100 * status.done) / status.total + "%" : "0%";
      var predRes = await fetch(LIVE + "predictions.json?t=" + (status.updated_at || ""));
      if (predRes.ok) {
        var body = await predRes.text();
        if (body !== lastPredKey) {
          lastPredKey = body;
          allPred = JSON.parse(body).predictions || [];
          updateLayers();
        }
      }
      stats.innerHTML =
        status.state.toUpperCase() +
        " · chips " +
        status.done +
        "/" +
        status.total +
        "<br><span style='color:#34d399'>" +
        allGt.length.toLocaleString() +
        " register</span> · <span style='color:#fbbf24'>" +
        allPred.length.toLocaleString() +
        " model</span>";
      if (status.state === "done" && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
        stats.innerHTML += "<br>Done — open ghent.html for final bundle.";
      }
    } catch (e) {
      stats.textContent = String(e);
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

  map.once("load", function () {
    updateLayers();
    poll();
    pollTimer = setInterval(poll, POLL_MS);
  });
})();
