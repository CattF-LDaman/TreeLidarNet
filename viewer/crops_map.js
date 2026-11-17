(function () {
  "use strict";

  var CENTER = [3.7252, 51.0543];
  var POLL_MS = 2000;
  var discoveryMinM = 2.0;
  var allGt = [];
  var allPred = [];
  var gtSpatialIndex = null;
  var discoveriesOnly = false;
  var selectedKey = null;
  var showGt = true;
  var showPred = true;
  var livePoll = true;
  var bevMin = 0.85;
  var verMin = 0;
  var pollTimer = null;
  var lastLiveKey = "";
  var cropCache = Object.create(null);
  var viewer = { setCrop: function () {} };

  var metaEl = document.getElementById("crop-meta");
  var statsEl = document.getElementById("map-stats");
  var liveEl = document.getElementById("live-status");
  var gpuStatus = document.getElementById("gpu-status");
  var canvas = document.getElementById("crop-canvas");
  var tip = document.getElementById("map-tip");
  var barBev = document.getElementById("bar-bev");
  var barVerV2 = document.getElementById("bar-ver-v2");
  var barVerV3 = document.getElementById("bar-ver-v3");
  var valBev = document.getElementById("val-bev");
  var valVerV2 = document.getElementById("val-ver-v2");
  var valVerV3 = document.getElementById("val-ver-v3");
  var verifierModeEl = document.getElementById("verifier-mode");
  var rowVerV2 = document.getElementById("row-ver-v2");
  var rowVerV3 = document.getElementById("row-ver-v3");

  function verifierMode() {
    return verifierModeEl ? verifierModeEl.value : "compare";
  }

  function haversineM(lon1, lat1, lon2, lat2) {
    var R = 6371000;
    var toRad = Math.PI / 180;
    var dLat = (lat2 - lat1) * toRad;
    var dLon = (lon2 - lon1) * toRad;
    var a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(lat1 * toRad) *
        Math.cos(lat2 * toRad) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
  }

  function buildGtSpatialIndex(gtList) {
    var cellDeg = 0.00008;
    var index = Object.create(null);
    for (var i = 0; i < gtList.length; i++) {
      var p = gtList[i].position;
      if (!p) continue;
      var cx = Math.floor(p[0] / cellDeg);
      var cy = Math.floor(p[1] / cellDeg);
      var key = cx + "," + cy;
      if (!index[key]) index[key] = [];
      index[key].push(p);
    }
    return { index: index, cellDeg: cellDeg };
  }

  function isNearRegister(lon, lat, minDistM) {
    if (!gtSpatialIndex || !allGt.length) return false;
    var cellDeg = gtSpatialIndex.cellDeg;
    var cx = Math.floor(lon / cellDeg);
    var cy = Math.floor(lat / cellDeg);
    var rings = Math.ceil(minDistM / (cellDeg * 111320)) + 1;
    for (var dx = -rings; dx <= rings; dx++) {
      for (var dy = -rings; dy <= rings; dy++) {
        var bucket = gtSpatialIndex.index[cx + dx + "," + (cy + dy)];
        if (!bucket) continue;
        for (var j = 0; j < bucket.length; j++) {
          var g = bucket[j];
          if (haversineM(lon, lat, g[0], g[1]) < minDistM) return true;
        }
      }
    }
    return false;
  }

  function predsForMap() {
    if (!discoveriesOnly) return allPred;
    var out = [];
    for (var i = 0; i < allPred.length; i++) {
      var d = allPred[i];
      if (!d.position) continue;
      if (!isNearRegister(d.position[0], d.position[1], discoveryMinM)) {
        out.push(d);
      }
    }
    return out;
  }

  var map = new maplibregl.Map({
    container: "map",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: CENTER,
    zoom: 13,
  });

  var overlay = new deck.MapboxOverlay({
    interleaved: true,
    layers: [],
    onClick: onMapClick,
    onHover: function (info) {
      if (!info.object || !info.layer) {
        tip.style.display = "none";
        return;
      }
      var d = info.object;
      var id = info.layer.id;
      tip.style.display = "block";
      tip.style.left = info.x + 14 + "px";
      tip.style.top = info.y + 14 + "px";
      if (id === "ghent-gt-trees") {
        tip.innerHTML =
          "<strong style='color:#34d399'>Register</strong><br>" +
          (d.sortiment || "") +
          (d.gis_id ? "<br><span style='color:#8b949e'>" + d.gis_id + "</span>" : "");
      } else if (id.indexOf("ghent-pred") === 0) {
        var bev = d.score != null ? d.score.toFixed(2) : "?";
        var v2 =
          d.verifier_scores && d.verifier_scores.v2 != null
            ? d.verifier_scores.v2.toFixed(2)
            : null;
        var v3 =
          d.verifier_scores && d.verifier_scores.v3 != null
            ? d.verifier_scores.v3.toFixed(2)
            : d.verifier_score != null
              ? d.verifier_score.toFixed(2)
              : null;
        var verLine = "(click for verifier)";
        if (v2 != null && v3 != null) {
          verLine = "v2 " + v2 + " · v3 " + v3;
        } else if (v3 != null) {
          verLine = "v3 " + v3;
        } else if (v2 != null) {
          verLine = "v2 " + v2;
        }
        tip.innerHTML =
          "<strong style='color:#fbbf24'>Detection</strong><br>BEV " +
          bev +
          "<br>Verifier " +
          verLine;
      } else {
        tip.style.display = "none";
      }
    },
  });
  map.addControl(overlay);

  map.once("load", function () {
    map.resize();
    loadRegister();
    if (livePoll) startLivePoll();
    else loadPredictions();
    updateLayers();
  });
  map.on("error", function (e) {
    console.error("map error", e);
    if (statsEl) statsEl.textContent = "Map failed to load (network or style).";
  });

  import("./crop_viewer_3d.js")
    .then(function (m) {
      return m.mountCropViewer(canvas, {
        onStatus: function (msg) {
          if (gpuStatus) gpuStatus.textContent = msg;
        },
      });
    })
    .then(function (v) {
      viewer = v;
    })
    .catch(function (err) {
      console.error(err);
      if (gpuStatus) gpuStatus.textContent = "3D viewer failed: " + (err.message || err);
    });

  function loadRegister() {
    if (statsEl) statsEl.textContent = "Loading register…";
    fetch("/api/register/all")
      .then(function (res) {
        if (!res.ok) throw new Error("register " + res.status);
        return res.json();
      })
      .then(function (data) {
        allGt = data.trees || [];
        gtSpatialIndex = buildGtSpatialIndex(allGt);
        updateLayers();
      })
      .catch(function (e) {
        console.error(e);
        if (statsEl) statsEl.textContent = "Register load failed: " + e.message;
      });
  }

  function loadPredictions() {
    if (livePoll) {
      return pollLivePredictions();
    }
    var path = document.getElementById("pred-path").value.trim();
    var q = path ? "?path=" + encodeURIComponent(path) : "";
    return fetch("/api/predictions" + q)
      .then(function (res) {
        if (!res.ok) throw new Error("predictions " + res.status);
        return res.json();
      })
      .then(function (data) {
        allPred = data.predictions || [];
        updateLayers();
      })
      .catch(function (e) {
        console.warn(e);
        allPred = [];
        updateLayers();
      });
  }

  function pollLivePredictions() {
    return Promise.all([
      fetch("/api/live/verified/status").then(function (r) { return r.json(); }),
      fetch("/api/live/verified/predictions").then(function (r) { return r.json(); }),
    ])
      .then(function (parts) {
        var st = parts[0];
        var body = parts[1];
        var key = (st.done || 0) + "/" + (st.total || 0) + st.phase;
        var cmpKey = body.compare ? ":cmp" : "";
        if (key + cmpKey !== lastLiveKey) {
          lastLiveKey = key + cmpKey;
          allPred = body.predictions || [];
          updateLayers();
        }
        if (liveEl) {
          if (st.phase === "scoring") {
            var cmpLive = st.compare ? " v2+v3" : "";
            liveEl.textContent =
              "Live scoring" + cmpLive + " " + (st.done || 0) + " / " + (st.total || 0) + "…";
            liveEl.style.color = "#fbbf24";
          } else if (st.phase === "done") {
            var cmp = st.compare ? " (v2+v3 compare)" : "";
            liveEl.textContent =
              "Live done" +
              cmp +
              " — " +
              (st.scored || st.done || 0) +
              " scored, " +
              (st.kept || "?") +
              " kept";
            liveEl.style.color = "#34d399";
          } else {
            liveEl.textContent = "Live idle — run filter-export --live";
            liveEl.style.color = "#8b949e";
          }
        }
      })
      .catch(function (e) {
        console.warn(e);
        if (liveEl) liveEl.textContent = "Live poll failed";
      });
  }

  function startLivePoll() {
    if (pollTimer) clearInterval(pollTimer);
    pollLivePredictions();
    pollTimer = setInterval(pollLivePredictions, POLL_MS);
  }

  function stopLivePoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    if (liveEl) liveEl.textContent = "Live: off";
  }

  function selectionLayer() {
    if (!selectedKey) return null;
    var isGt = selectedKey.type === "gt";
    var list = isGt ? allGt : predsForMap();
    var hit = null;
    for (var i = 0; i < list.length; i++) {
      var d = list[i];
      if (d.gis_id === selectedKey.id || d.__key === selectedKey.id) {
        hit = d;
        break;
      }
    }
    if (!hit) return null;
    return new deck.ScatterplotLayer({
      id: "selection-halo",
      data: [hit],
      pickable: false,
      filled: false,
      stroked: true,
      radiusUnits: "meters",
      getPosition: function (d) {
        return d.position;
      },
      getRadius: function () {
        return isGt ? (hit.radiusMeters || 3) * 1.35 : 8;
      },
      getLineColor: [147, 197, 253, 255],
      getLineWidth: 4,
      lineWidthUnits: "pixels",
    });
  }

  function visiblePredCount() {
    var n = 0;
    var mode = verifierMode();
    var list = predsForMap();
    for (var i = 0; i < list.length; i++) {
      var d = list[i];
      var bev = d.score != null ? d.score : 0;
      if (bev < bevMin) continue;
      if (verMin > 0) {
        var ver =
          TreeLidarMap.verForRow && TreeLidarMap.verForRow(d, mode === "v2" ? "v2" : "v3");
        if (ver == null && mode === "compare" && d.verifier_score != null) {
          ver = d.verifier_score;
        }
        if (ver == null && mode === "compare" && d.verifier_scores) {
          ver = d.verifier_scores.v3 != null ? d.verifier_scores.v3 : d.verifier_scores.v2;
        }
        if (ver != null && ver < verMin) continue;
      }
      n++;
    }
    return n;
  }

  function updateLayers() {
    var preds = predsForMap();
    var base = TreeLidarMap.buildLayers(allGt, preds, {
      showGt: showGt,
      showPred: showPred,
      showPredHeat: false,
      bevMin: bevMin,
      verifierMin: verMin,
      haloVerifier: verifierMode(),
    });
    var extra = selectionLayer();
    overlay.setProps({ layers: extra ? base.concat(extra) : base });
    if (statsEl) {
      var vis = visiblePredCount();
      var basePred = preds.length.toLocaleString();
      var discNote = discoveriesOnly
        ? " discoveries (>" + discoveryMinM.toFixed(1) + " m from register)"
        : " preds";
      statsEl.textContent =
        allGt.length.toLocaleString() +
        " register · " +
        vis.toLocaleString() +
        " / " +
        basePred +
        discNote +
        (discoveriesOnly ? " · " + allPred.length.toLocaleString() + " total" : " (filtered)");
    }
  }

  function setBar(barEl, valEl, val) {
    if (!barEl || !valEl) return;
    if (val != null && !isNaN(val)) {
      barEl.style.width = Math.round(Math.min(1, Math.max(0, val)) * 100) + "%";
      valEl.textContent = val.toFixed(2);
    } else {
      barEl.style.width = "0%";
      valEl.textContent = "—";
    }
  }

  function setScoreBars(bev, scores) {
    var mode = verifierMode();
    setBar(barBev, valBev, bev);
    var v2 = scores && scores.v2 != null ? scores.v2 : null;
    var v3 = scores && scores.v3 != null ? scores.v3 : null;
    if (rowVerV2) rowVerV2.style.display = mode === "v3" ? "none" : "flex";
    if (rowVerV3) rowVerV3.style.display = mode === "v2" ? "none" : "flex";
    setBar(barVerV2, valVerV2, mode === "v3" ? null : v2);
    setBar(barVerV3, valVerV3, mode === "v2" ? null : v3);
  }

  function onPick(info, event) {
    if (!info || !info.object) return;
    var d = info.object;
    if (d.gis_id) {
      selectedKey = { type: "gt", id: d.gis_id };
      loadCrop({ gis_id: d.gis_id }, d);
    } else if (d.position) {
      var key = d.position[0] + "," + d.position[1];
      d.__key = key;
      selectedKey = { type: "pred", id: key };
      loadCropWgs84(d.position[0], d.position[1], d.score, d.verifier_score, d);
    }
    updateLayers();
    if (event && event.stopPropagation) event.stopPropagation();
  }

  function onMapClick(info) {
    onPick(info);
  }

  function loadCrop(params, row) {
    var key =
      verifierMode() + ":" + (params.gis_id || params.x + "," + params.y);
    params.verifier_mode = verifierMode();
    if (cropCache[key]) {
      applyCrop(cropCache[key], row);
      return;
    }
    metaEl.textContent = "Loading crop…";
    var qs = new URLSearchParams(params);
    fetch("/api/crop?" + qs)
      .then(function (res) {
        if (!res.ok) throw new Error(String(res.status));
        return res.json();
      })
      .then(function (payload) {
        cropCache[key] = payload;
        applyCrop(payload, row);
      })
      .catch(function () {
        metaEl.textContent = "Crop failed — chip may have no LiDAR here.";
        viewer.setCrop(null);
        setScoreBars(null, null);
      });
  }

  function loadCropWgs84(lon, lat, score, verifierScore, row) {
    var key = verifierMode() + ":wgs:" + lon + "," + lat;
    if (cropCache[key]) {
      applyCrop(cropCache[key], row);
      return;
    }
    metaEl.textContent = "Loading crop…";
    var qs = new URLSearchParams({ lon: String(lon), lat: String(lat), source: "prediction" });
    qs.set("verifier_mode", verifierMode());
    if (score != null) qs.set("score", String(score));
    if (verifierScore != null) qs.set("verifier_score", String(verifierScore));
    fetch("/api/crop/wgs84?" + qs)
      .then(function (res) {
        if (!res.ok) throw new Error(String(res.status));
        return res.json();
      })
      .then(function (payload) {
        cropCache[key] = payload;
        applyCrop(payload, row);
      })
      .catch(function () {
        metaEl.textContent = "Crop failed — chip may have no LiDAR here.";
        viewer.setCrop(null);
        setScoreBars(null, null);
      });
  }

  function applyCrop(payload, row) {
    viewer.setCrop(payload);
    var gis = payload.gis_id || (row && row.gis_id) || "—";
    var sortiment = row && row.sortiment ? row.sortiment : "—";
    var bev =
      payload.score != null
        ? payload.score
        : row && row.score != null
          ? row.score
          : null;
    var scores = payload.verifier_scores || {};
    if (payload.verifier_score != null && Object.keys(scores).length === 0) {
      scores = { v3: payload.verifier_score };
    }
    setScoreBars(bev, scores);
    var rgbNote =
      payload.points[0] && payload.points[0].red != null
        ? " · RGB in crop"
        : " · no RGB (v3 may fail)";
    metaEl.innerHTML =
      "<b>" +
      gis +
      "</b> · " +
      sortiment +
      "<br>chip " +
      payload.chip_id +
      " · " +
      payload.points.length +
      " pts · r=" +
      payload.radius_m +
      " m" +
      rgbNote;
  }

  document.getElementById("show-register").addEventListener("change", function (e) {
    showGt = e.target.checked;
    updateLayers();
  });
  document.getElementById("show-preds").addEventListener("change", function (e) {
    showPred = e.target.checked;
    updateLayers();
  });
  var discoveriesEl = document.getElementById("discoveries-only");
  var discoverySliderWrap = document.getElementById("discovery-slider-wrap");
  var discoveryMinEl = document.getElementById("discovery-min-m");
  function syncDiscoverySliderUi() {
    if (discoverySliderWrap) {
      discoverySliderWrap.style.display = discoveriesOnly ? "block" : "none";
    }
  }
  if (discoveriesEl) {
    discoveriesEl.addEventListener("change", function (e) {
      discoveriesOnly = e.target.checked;
      syncDiscoverySliderUi();
      updateLayers();
    });
  }
  if (discoveryMinEl) {
    discoveryMinEl.addEventListener("input", function (e) {
      discoveryMinM = parseFloat(e.target.value);
      var lab = document.getElementById("discovery-min-label");
      if (lab) lab.textContent = discoveryMinM.toFixed(1);
      if (discoveriesOnly) updateLayers();
    });
  }
  syncDiscoverySliderUi();
  document.getElementById("pred-path").addEventListener("change", function () {
    if (!livePoll) loadPredictions();
  });
  document.getElementById("live-poll").addEventListener("change", function (e) {
    livePoll = e.target.checked;
    if (livePoll) startLivePoll();
    else stopLivePoll();
  });
  document.getElementById("bev-min").addEventListener("input", function (e) {
    bevMin = parseFloat(e.target.value);
    document.getElementById("bev-min-label").textContent = bevMin.toFixed(2);
    updateLayers();
  });
  document.getElementById("ver-min").addEventListener("input", function (e) {
    verMin = parseFloat(e.target.value);
    document.getElementById("ver-min-label").textContent = verMin.toFixed(2);
    updateLayers();
  });
  if (verifierModeEl) {
    verifierModeEl.addEventListener("change", function () {
      cropCache = Object.create(null);
      updateLayers();
      if (selectedKey && statsEl) {
        statsEl.textContent += " · re-click for crop panel";
      }
    });
  }
})();