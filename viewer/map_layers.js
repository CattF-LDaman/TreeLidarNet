/**
 * Map layers: LiDAR model (amber rings) is the product; Gent register (green) is optional overlay.
 */
window.TreeLidarMap = (function () {
  var GREEN = [52, 211, 153];
  var GREEN_DIM = [101, 163, 122];
  var PRED_RING = [251, 191, 36];
  var PRED_RING_GLOW = [245, 158, 11];

  function gtFill(d) {
    var base = d.hasSize ? GREEN : GREEN_DIM;
    var alpha = d.hasSize ? 175 : 168;
    return [base[0], base[1], base[2], alpha];
  }

  function gtLine(d) {
    return d.hasSize ? [167, 243, 208, 80] : [167, 209, 181, 185];
  }

  function gtRadius(d) {
    return d.radiusMeters || 3;
  }

  function buildLayers(allGt, allPred, opts) {
    var showGt = opts.showGt !== false;
    var showPred = opts.showPred !== false;
    var showPredHeat = !!opts.showPredHeat;
    var layers = [];

    if (showPredHeat && allPred.length) {
      layers.push(
        new deck.HeatmapLayer({
          id: "pred-heat",
          data: allPred,
          pickable: false,
          getPosition: function (d) {
            return d.position;
          },
          getWeight: function (d) {
            return d.score || 0.5;
          },
          radiusPixels: 56,
          intensity: 2.2,
          threshold: 0.03,
        })
      );
    }

    /* Model predictions underneath — hollow rings so register stays readable on top. */
    if (showPred && allPred.length) {
      layers.push(
        new deck.ScatterplotLayer({
          id: "ghent-pred-rings",
          data: allPred,
          pickable: true,
          filled: false,
          stroked: true,
          radiusUnits: "meters",
          getPosition: function (d) {
            return d.position;
          },
          getRadius: function () {
            return 5.5;
          },
          getLineColor: function (d) {
            var s = d.score || 0;
            if (s >= 0.6) return [PRED_RING[0], PRED_RING[1], PRED_RING[2], 240];
            return [PRED_RING_GLOW[0], PRED_RING_GLOW[1], PRED_RING_GLOW[2], 200];
          },
          getLineWidth: 3,
          lineWidthUnits: "pixels",
          lineWidthMinPixels: 2,
          lineWidthMaxPixels: 4,
        })
      );
    }

    if (showGt && allGt.length) {
      layers.push(
        new deck.ScatterplotLayer({
          id: "ghent-gt-trees",
          data: allGt,
          pickable: true,
          filled: true,
          stroked: true,
          radiusUnits: "meters",
          getPosition: function (d) {
            return d.position;
          },
          getRadius: gtRadius,
          getFillColor: gtFill,
          getLineColor: gtLine,
          getLineWidth: function (d) {
            return d.hasSize ? 1 : 1.5;
          },
          lineWidthUnits: "pixels",
          lineWidthMinPixels: 1,
          autoHighlight: true,
          highlightColor: [167, 243, 208, 220],
        })
      );
    }

    return layers;
  }

  function syncOverlay(overlay, allGt, allPred, opts) {
    var showPredHeat = !!opts.showPredHeat;
    overlay.setProps({
      interleaved: !showPredHeat,
      layers: buildLayers(allGt, allPred, opts),
    });
  }

  return {
    buildLayers,
    syncOverlay,
    LEGEND: {
      gt: "Gent register (open data) — filled green",
      pred: "LiDAR model — amber ring (under green)",
    },
  };
})();
