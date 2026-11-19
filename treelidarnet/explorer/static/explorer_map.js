(function () {
  "use strict";

  var CENTER = [3.7252, 51.0543];
  var DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

  var map = null;
  var overlay = null;
  var mapReady = false;
  var bundle = null;
  var styleCfg = null;

  function fallbackStyle() {
    return {
      repr: "ring",
      colorField: "v3",
      sizeField: "none",
      baseRadiusM: 5.5,
      sizeBoostM: 4,
      lineWidthPx: 3,
      opacity: 0.88,
      hueLow: 8,
      hueHigh: 145,
      sat: 0.82,
      heatmap: false,
    };
  }

  function ensureStyle() {
    if (!styleCfg) {
      styleCfg = window.ExplorerStyle ? ExplorerStyle.defaultStyle() : fallbackStyle();
    }
    return styleCfg;
  }
  var visibleLon = new Float32Array(0);
  var visibleLat = new Float32Array(0);
  var visibleBev = new Float32Array(0);
  var visibleV2 = new Float32Array(0);
  var visibleV3 = new Float32Array(0);
  var visibleLineColor = new Uint8ClampedArray(0);
  var visibleFillColor = new Uint8ClampedArray(0);
  var visibleRadius = new Float32Array(0);
  var visibleHeatW = new Float32Array(0);
  var visibleHaloOuter = null;
  var visibleHaloInner = null;
  var visibleCount = 0;
  var showRegister = true;
  var hideFlagged = false;
  var lastFilterCfg = null;
  var scatterPositionsBuf = new Float32Array(0);

  function recomputeAppearance() {
    if (!visibleCount) return;
    var style = ensureStyle();
    if (!window.ExplorerStyle) {
      visibleLineColor = new Uint8ClampedArray(visibleCount * 4);
      visibleFillColor = new Uint8ClampedArray(visibleCount * 4);
      visibleRadius = new Float32Array(visibleCount);
      visibleHeatW = new Float32Array(visibleCount);
      for (var i = 0; i < visibleCount; i++) {
        visibleRadius[i] = style.baseRadiusM;
        visibleHeatW[i] = visibleV3[i];
        var o = i * 4;
        visibleLineColor[o] = 251;
        visibleLineColor[o + 1] = 191;
        visibleLineColor[o + 2] = 36;
        visibleLineColor[o + 3] = Math.round(style.opacity * 220);
      }
      return;
    }
    var app = ExplorerStyle.buildAppearance(
      visibleCount,
      visibleBev,
      visibleV2,
      visibleV3,
      style
    );
    visibleLineColor = app.lineColor;
    visibleFillColor = app.fillColor;
    visibleRadius = app.radiusM;
    visibleHeatW = app.heatWeight;
    visibleHaloOuter = app.haloOuter;
    visibleHaloInner = app.haloInner;
  }

  function compactVisible(cols, mask) {
    var n = mask.length;
    var skipFlagged = hideFlagged && window.ExplorerFlags;
    var kept = 0;
    for (var i = 0; i < n; i++) {
      if (!mask[i]) continue;
      if (skipFlagged && ExplorerFlags.isFlagged(cols.lon[i], cols.lat[i])) continue;
      kept++;
    }
    visibleLon = new Float32Array(kept);
    visibleLat = new Float32Array(kept);
    visibleBev = new Float32Array(kept);
    visibleV2 = new Float32Array(kept);
    visibleV3 = new Float32Array(kept);
    scatterPositionsBuf = new Float32Array(kept * 2);
    var j = 0;
    for (var k = 0; k < n; k++) {
      if (!mask[k]) continue;
      if (skipFlagged && ExplorerFlags.isFlagged(cols.lon[k], cols.lat[k])) continue;
      visibleLon[j] = cols.lon[k];
      visibleLat[j] = cols.lat[k];
      visibleBev[j] = cols.bev[k];
      visibleV2[j] = cols.v2[k];
      visibleV3[j] = cols.v3[k];
      scatterPositionsBuf[j * 2] = cols.lon[k];
      scatterPositionsBuf[j * 2 + 1] = cols.lat[k];
      j++;
    }
    visibleCount = kept;
    recomputeAppearance();
    return kept;
  }

  function predLayers(zoom) {
    var layers = [];
    var style = ensureStyle();
    var useHeat = style.heatmap && zoom < 12 && visibleCount > 0;
    var repr = style.repr;

    if (useHeat) {
      var maxPts = 80000;
      var step = Math.max(1, Math.ceil(visibleCount / maxPts));
      var heatData = [];
      var i;
      for (i = 0; i < visibleCount; i += step) {
        heatData.push({
          position: [visibleLon[i], visibleLat[i]],
          weight: visibleHeatW[i] * visibleBev[i],
        });
      }
      layers.push(
        new deck.HeatmapLayer({
          id: "pred-heat",
          data: heatData,
          getPosition: function (d) {
            return d.position;
          },
          getWeight: function (d) {
            return d.weight;
          },
          radiusPixels: 40,
          intensity: 2,
          threshold: 0.03,
        })
      );
      return layers;
    }

    if (visibleCount === 0) return layers;

    if (repr === "compare_halos") {
      var outerR = visibleHaloOuter || visibleRadius;
      var innerR = visibleHaloInner || visibleRadius;
      layers.push(
        new deck.ScatterplotLayer({
          id: "pred-halo-v2",
          data: {
            length: visibleCount,
            attributes: {
              getPosition: { value: scatterPositionsBuf, size: 2 },
              getFillColor: { value: visibleLineColor, size: 4 },
              getRadius: { value: outerR, size: 1 },
            },
          },
          radiusUnits: "meters",
          filled: true,
          stroked: false,
          pickable: false,
        })
      );
      layers.push(
        new deck.ScatterplotLayer({
          id: "pred-halo-v3",
          data: {
            length: visibleCount,
            attributes: {
              getPosition: { value: scatterPositionsBuf, size: 2 },
              getFillColor: { value: visibleFillColor, size: 4 },
              getRadius: { value: innerR, size: 1 },
            },
          },
          radiusUnits: "meters",
          filled: true,
          stroked: false,
          pickable: true,
          autoHighlight: true,
          highlightColor: [251, 191, 36, 200],
        })
      );
      return layers;
    }

    if (repr === "filled" || repr === "halo_v3") {
      layers.push(
        new deck.ScatterplotLayer({
          id: "pred-filled",
          data: {
            length: visibleCount,
            attributes: {
              getPosition: { value: scatterPositionsBuf, size: 2 },
              getFillColor: { value: visibleFillColor, size: 4 },
              getRadius: { value: visibleRadius, size: 1 },
            },
          },
          radiusUnits: "meters",
          filled: true,
          stroked: false,
          pickable: true,
          autoHighlight: true,
          highlightColor: [251, 191, 36, 220],
        })
      );
      if (repr === "halo_v3") {
        layers.push(
          new deck.ScatterplotLayer({
            id: "pred-ring-accent",
            data: {
              length: visibleCount,
              attributes: {
                getPosition: { value: scatterPositionsBuf, size: 2 },
                getLineColor: { value: visibleLineColor, size: 4 },
                getRadius: { value: visibleRadius, size: 1 },
              },
            },
            stroked: true,
            filled: false,
            radiusUnits: "meters",
            lineWidthUnits: "pixels",
            getLineWidth: style.lineWidthPx,
            pickable: false,
          })
        );
      }
      return layers;
    }

    layers.push(
      new deck.ScatterplotLayer({
        id: "pred-scatter",
        data: {
          length: visibleCount,
          attributes: {
            getPosition: { value: scatterPositionsBuf, size: 2 },
            getLineColor: { value: visibleLineColor, size: 4 },
            getRadius: { value: visibleRadius, size: 1 },
          },
        },
        radiusUnits: "meters",
        stroked: true,
        filled: false,
        lineWidthUnits: "pixels",
        lineWidthMinPixels: 1,
        lineWidthMaxPixels: 10,
        getLineWidth: style.lineWidthPx,
        pickable: true,
        autoHighlight: true,
        highlightColor: [251, 191, 36, 220],
      })
    );
    return layers;
  }

  function buildLayers(zoom) {
    var layers = predLayers(zoom);
    if (window.ExplorerFlags) {
      var markers = ExplorerFlags.flagMarkers();
      if (markers.length) {
        layers.push(
          new deck.ScatterplotLayer({
            id: "qa-flags",
            data: markers,
            getPosition: function (d) {
              return d.position;
            },
            getRadius: 5,
            radiusUnits: "meters",
            getFillColor: [248, 81, 73, 220],
            stroked: true,
            getLineColor: [255, 255, 255, 180],
            lineWidthMinPixels: 2,
            pickable: false,
          })
        );
      }
    }
    if (showRegister && bundle && bundle.register.length) {
      layers.push(
        new deck.ScatterplotLayer({
          id: "register",
          data: bundle.register,
          getPosition: function (d) {
            return d.position;
          },
          getRadius: function (d) {
            return d.radiusMeters || 3;
          },
          radiusUnits: "meters",
          getFillColor: function (d) {
            return d.hasSize ? [52, 211, 153, 170] : [101, 163, 122, 150];
          },
          stroked: true,
          getLineColor: [167, 243, 208, 60],
          lineWidthMinPixels: 1,
          pickable: false,
        })
      );
    }
    return layers;
  }

  function refreshDeck() {
    if (!overlay || !map || !mapReady) return;
    var zoom = map.getZoom();
    overlay.setProps({ layers: buildLayers(zoom) });
  }

  function applyFilter(cfg, cb) {
    if (!bundle) {
      cb && cb(0);
      return;
    }
    lastFilterCfg = cfg;
    var res = ExplorerCombine.filterMask(bundle.detections, cfg);
    var kept = compactVisible(bundle.detections, res.mask);
    refreshDeck();
    cb && cb(kept);
  }

  function initMap() {
    var loadStatus = document.getElementById("load-status");
    map = new maplibregl.Map({
      container: "explorer-map",
      style: DARK_STYLE,
      center: CENTER,
      zoom: 13,
      pitch: 0,
    });
    overlay = new deck.MapboxOverlay({ interleaved: !ensureStyle().heatmap, layers: [] });
    map.addControl(overlay);
    map.on("move", refreshDeck);
    map.on("zoom", refreshDeck);
    map.on("click", function (ev) {
      window.ExplorerDetail && window.ExplorerDetail.onMapClick(ev.lngLat);
    });
    map.on("error", function (e) {
      console.error("map error", e);
      if (loadStatus) loadStatus.textContent = "Map style failed — check network.";
    });
    map.once("load", function () {
      mapReady = true;
      map.resize();
      refreshDeck();
      if (loadStatus && loadStatus.textContent.indexOf("Loaded") === -1) {
        loadStatus.textContent = "Map ready — loading detections…";
      }
    });
  }

  window.ExplorerMap = {
    init: initMap,
    setBundle: function (b) {
      bundle = b;
      refreshDeck();
    },
    applyFilter: applyFilter,
    setStyle: function (style) {
      var base = window.ExplorerStyle ? ExplorerStyle.defaultStyle() : fallbackStyle();
      styleCfg = Object.assign(base, style || {});
      if (overlay) {
        overlay.setProps({ interleaved: !styleCfg.heatmap });
      }
      recomputeAppearance();
      refreshDeck();
    },
    setShowRegister: function (v) {
      showRegister = v;
      refreshDeck();
    },
    setHideFlagged: function (v) {
      hideFlagged = v;
      if (lastFilterCfg) {
        applyFilter(lastFilterCfg);
      } else {
        refreshDeck();
      }
    },
    onFlagsChanged: function () {
      if (lastFilterCfg) {
        applyFilter(lastFilterCfg, function () {});
      } else {
        refreshDeck();
      }
    },
    nearestIndex: function (lon, lat, maxM) {
      var best = -1;
      var bestD = maxM || 12;
      for (var i = 0; i < visibleCount; i++) {
        var dLon = (visibleLon[i] - lon) * 111320 * Math.cos((lat * Math.PI) / 180);
        var dLat = (visibleLat[i] - lat) * 110540;
        var d = Math.sqrt(dLon * dLon + dLat * dLat);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      return best;
    },
    rowAt: function (idx) {
      if (idx < 0) return null;
      return {
        lon: visibleLon[idx],
        lat: visibleLat[idx],
        bev: visibleBev[idx],
        v2: visibleV2[idx],
        v3: visibleV3[idx],
      };
    },
  };
})();