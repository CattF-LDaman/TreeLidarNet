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

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function verForRow(d, key) {
    if (d.verifier_scores && d.verifier_scores[key] != null) {
      return d.verifier_scores[key];
    }
    if (key === "v3" && d.verifier_score != null && !d.verifier_scores) {
      return d.verifier_score;
    }
    return null;
  }

  function activeVerifier(d, opts) {
    var mode = opts.haloVerifier || "compare";
    if (mode === "v2") return verForRow(d, "v2");
    if (mode === "v3") return verForRow(d, "v3");
    var v3 = verForRow(d, "v3");
    if (v3 != null) return v3;
    return verForRow(d, "v2");
  }

  function haloColor(v) {
    var t = Math.min(1, Math.max(0, v));
    return [
      Math.round(lerp(248, 52, t)),
      Math.round(lerp(113, 211, t)),
      Math.round(lerp(133, 153, t)),
      Math.round(lerp(30, 90, t)),
    ];
  }

  function haloColorV2(v) {
    var t = Math.min(1, Math.max(0, v));
    return [
      Math.round(lerp(248, 96, t)),
      Math.round(lerp(113, 165, t)),
      Math.round(lerp(248, 250, t)),
      Math.round(lerp(28, 88, t)),
    ];
  }

  function predRingStyle(d, opts) {
    var bev = d.score != null ? d.score : 0;
    var ver = activeVerifier(d, opts);
    var alpha = Math.round(lerp(140, 255, Math.min(1, Math.max(0, (bev - 0.2) / 0.8))));
    if (ver == null) {
      if (bev >= 0.9) {
        return { color: [251, 191, 36, alpha], width: 4 };
      }
      if (bev >= 0.6) {
        return { color: [245, 158, 11, alpha], width: 3 };
      }
      return { color: [180, 120, 40, Math.min(200, alpha)], width: 2 };
    }
    var t = Math.min(1, Math.max(0, ver));
    var r = Math.round(lerp(251, 52, t));
    var g = Math.round(lerp(113, 211, t));
    var b = Math.round(lerp(133, 153, t));
    var w = 2 + Math.round(3 * Math.min(1, bev));
    if (ver < 0.35) {
      return { color: [248, 113, 113, 230], width: w + 1 };
    }
    return { color: [r, g, b, alpha], width: w };
  }

  function hasDualScores(allPred) {
    return allPred.some(function (d) {
      return d.verifier_scores && d.verifier_scores.v2 != null && d.verifier_scores.v3 != null;
    });
  }

  function buildLayers(allGt, allPred, opts) {
    var showGt = opts.showGt !== false;
    var showPred = opts.showPred !== false;
    var showPredHeat = !!opts.showPredHeat;
    var bevMin = opts.bevMin != null ? opts.bevMin : 0;
    var verMin = opts.verifierMin != null ? opts.verifierMin : 0;
    var haloMode = opts.haloVerifier || "compare";
    var dual = hasDualScores(allPred);
    if (allPred.length && (bevMin > 0 || verMin > 0)) {
      allPred = allPred.filter(function (d) {
        var bev = d.score != null ? d.score : 0;
        if (bev < bevMin) return false;
        if (verMin > 0) {
          var ver = activeVerifier(d, { haloVerifier: haloMode });
          if (ver != null && ver < verMin) return false;
        }
        return true;
      });
    }
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

    /* Amber rings last, so a detection draws over the green disc it lands on. */
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
            return predRingStyle(d, opts).color;
          },
          getLineWidth: function (d) {
            return predRingStyle(d, opts).width;
          },
          lineWidthUnits: "pixels",
          lineWidthMinPixels: 2,
          lineWidthMaxPixels: 6,
        })
      );

      function addHaloLayer(id, key, colorFn, radiusBase) {
        var data = allPred.filter(function (d) {
          return verForRow(d, key) != null;
        });
        if (!data.length) return;
        layers.push(
          new deck.ScatterplotLayer({
            id: id,
            data: data,
            pickable: false,
            filled: true,
            stroked: false,
            radiusUnits: "meters",
            getPosition: function (d) {
              return d.position;
            },
            getRadius: function (d) {
              var v = verForRow(d, key) || 0;
              return radiusBase + v * 4;
            },
            getFillColor: function (d) {
              return colorFn(verForRow(d, key) || 0);
            },
          })
        );
      }

      if (dual && haloMode === "compare") {
        addHaloLayer("ghent-pred-verifier-halo-v2", "v2", haloColorV2, 7);
        addHaloLayer("ghent-pred-verifier-halo-v3", "v3", haloColor, 3);
      } else if (haloMode === "v2") {
        addHaloLayer("ghent-pred-verifier-halo-v2", "v2", haloColorV2, 4);
      } else if (haloMode === "v3" || allPred.some(function (d) { return d.verifier_score != null; })) {
        addHaloLayer("ghent-pred-verifier-halo-v3", "v3", haloColor, 3);
      }
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
    verForRow,
    LEGEND: {
      gt: "Gent register (open data) — filled green",
      pred: "LiDAR model — amber ring (on top of green)",
    },
  };
})();