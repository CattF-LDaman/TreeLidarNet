/* Detection appearance: color/size encoding and ramps (shared by map + style UI). */
(function (global) {
  "use strict";

  function clamp01(v) {
    return Math.min(1, Math.max(0, v));
  }

  function hslToRgb(h, s, l) {
    h = ((h % 360) + 360) % 360;
    s = clamp01(s);
    l = clamp01(l);
    var c = (1 - Math.abs(2 * l - 1)) * s;
    var x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    var m = l - c / 2;
    var r = 0;
    var g = 0;
    var b = 0;
    if (h < 60) {
      r = c;
      g = x;
    } else if (h < 120) {
      r = x;
      g = c;
    } else if (h < 180) {
      g = c;
      b = x;
    } else if (h < 240) {
      g = x;
      b = c;
    } else if (h < 300) {
      r = x;
      b = c;
    } else {
      r = c;
      b = x;
    }
    return [
      Math.round((r + m) * 255),
      Math.round((g + m) * 255),
      Math.round((b + m) * 255),
    ];
  }

  function defaultStyle() {
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

  function rawScoreForField(bev, v2, v3, field) {
    switch (field) {
      case "bev":
        return bev;
      case "v2":
        return v2;
      case "v3":
        return v3;
      case "max_v":
        return Math.max(v2, v3);
      case "min_v":
        return Math.min(v2, v3);
      case "disagree":
        return Math.abs(v2 - v3);
      case "blend":
        return 0.5 * v2 + 0.5 * v3;
      default:
        return v3;
    }
  }

  /** Min/max over visible detections only (used for color + size ramps). */
  function rangeForField(count, bev, v2, v3, field) {
    if (!field || field === "none" || count === 0) {
      return { min: 0, max: 1 };
    }
    var min = Infinity;
    var max = -Infinity;
    var i;
    for (i = 0; i < count; i++) {
      var v = rawScoreForField(bev[i], v2[i], v3[i], field);
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!isFinite(min)) return { min: 0, max: 1 };
    if (max <= min) max = min + 1e-9;
    return { min: min, max: max };
  }

  function normFromRange(raw, range) {
    return clamp01((raw - range.min) / (range.max - range.min));
  }

  function colorRgba(bev, v2, v3, style, colorRange) {
    var raw = rawScoreForField(bev, v2, v3, style.colorField);
    var t = colorRange
      ? normFromRange(raw, colorRange)
      : clamp01(raw);
    var hue = style.hueLow + (style.hueHigh - style.hueLow) * t;
    var rgb = hslToRgb(hue, style.sat, 0.52);
    var a = Math.round(style.opacity * 255);
    return [rgb[0], rgb[1], rgb[2], a];
  }

  function v2HaloColor(tNorm, opacity) {
    var t = clamp01(tNorm);
    return [
      Math.round(248 - t * 152),
      Math.round(113 + t * 82),
      Math.round(248 - t * 2),
      Math.round(opacity * 0.45 * 255),
    ];
  }

  function v3HaloColor(tNorm, opacity) {
    var t = clamp01(tNorm);
    return [
      Math.round(248 - t * 196),
      Math.round(113 + t * 98),
      Math.round(133 + t * 20),
      Math.round(opacity * 0.55 * 255),
    ];
  }

  function radiusM(tNorm, style) {
    var base = style.baseRadiusM;
    var field = style.sizeField;
    if (field === "none" || !field) return base;
    return base + clamp01(tNorm) * style.sizeBoostM;
  }

  function buildAppearance(count, bev, v2, v3, style) {
    var line = new Uint8ClampedArray(count * 4);
    var fill = new Uint8ClampedArray(count * 4);
    var radii = new Float32Array(count);
    var heatWeight = new Float32Array(count);
    var haloOuter = style.repr === "compare_halos" ? new Float32Array(count) : null;
    var haloInner = style.repr === "compare_halos" ? new Float32Array(count) : null;

    var colorRange = rangeForField(count, bev, v2, v3, style.colorField);
    var sizeField = style.sizeField === "none" || !style.sizeField ? null : style.sizeField;
    var sizeRange = sizeField ? rangeForField(count, bev, v2, v3, sizeField) : null;
    var v2Range = rangeForField(count, bev, v2, v3, "v2");
    var v3Range = rangeForField(count, bev, v2, v3, "v3");

    var i;
    for (i = 0; i < count; i++) {
      var rgba = colorRgba(bev[i], v2[i], v3[i], style, colorRange);
      var o = i * 4;
      line[o] = rgba[0];
      line[o + 1] = rgba[1];
      line[o + 2] = rgba[2];
      line[o + 3] = rgba[3];

      var colorRaw = rawScoreForField(bev[i], v2[i], v3[i], style.colorField);
      heatWeight[i] = normFromRange(colorRaw, colorRange);

      var sizeNorm = 0;
      if (sizeRange) {
        var sizeRaw = rawScoreForField(bev[i], v2[i], v3[i], sizeField);
        sizeNorm = normFromRange(sizeRaw, sizeRange);
      }
      radii[i] = radiusM(sizeNorm, style);

      var nV2 = normFromRange(v2[i], v2Range);
      var nV3 = normFromRange(v3[i], v3Range);

      if (style.repr === "filled" || style.repr === "halo_v3") {
        fill[o] = rgba[0];
        fill[o + 1] = rgba[1];
        fill[o + 2] = rgba[2];
        fill[o + 3] = Math.round(rgba[3] * 0.65);
      } else if (style.repr === "compare_halos") {
        var h3 = v3HaloColor(nV3, style.opacity);
        fill[o] = h3[0];
        fill[o + 1] = h3[1];
        fill[o + 2] = h3[2];
        fill[o + 3] = h3[3];
        var h2 = v2HaloColor(nV2, style.opacity);
        line[o] = h2[0];
        line[o + 1] = h2[1];
        line[o + 2] = h2[2];
        line[o + 3] = Math.round(style.opacity * 200);
        haloOuter[i] = Math.max(7, radii[i] + 2 + nV2 * (style.sizeBoostM + 2));
        haloInner[i] = Math.max(3, radii[i] * 0.55 + nV3 * style.sizeBoostM);
      } else {
        fill[o + 3] = 0;
      }
    }
    return {
      lineColor: line,
      fillColor: fill,
      radiusM: radii,
      heatWeight: heatWeight,
      haloOuter: haloOuter,
      haloInner: haloInner,
      ranges: {
        color: colorRange,
        size: sizeRange,
      },
    };
  }

  global.ExplorerStyle = {
    defaultStyle: defaultStyle,
    rawScoreForField: rawScoreForField,
    rangeForField: rangeForField,
    buildAppearance: buildAppearance,
    colorRgba: colorRgba,
  };
})(window);