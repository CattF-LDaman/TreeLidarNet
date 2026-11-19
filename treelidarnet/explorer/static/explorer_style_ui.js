(function () {
  "use strict";

  var STORAGE_KEY = "treelidar.explorer.style";

  function el(id) {
    return document.getElementById(id);
  }

  function readStyle() {
    return {
      repr: el("style-repr").value,
      colorField: el("style-color-field").value,
      sizeField: el("style-size-field").value,
      baseRadiusM: parseFloat(el("style-base-radius").value),
      sizeBoostM: parseFloat(el("style-size-boost").value),
      lineWidthPx: parseFloat(el("style-line-width").value),
      opacity: parseFloat(el("style-opacity").value),
      hueLow: parseFloat(el("style-hue-low").value),
      hueHigh: parseFloat(el("style-hue-high").value),
      sat: parseFloat(el("style-sat").value),
      heatmap: el("style-heatmap").checked,
    };
  }

  function syncLabels() {
    el("style-base-radius-label").textContent = el("style-base-radius").value;
    el("style-size-boost-label").textContent = el("style-size-boost").value;
    el("style-line-width-label").textContent = el("style-line-width").value;
    el("style-opacity-label").textContent = el("style-opacity").value;
    el("style-hue-low-label").textContent = el("style-hue-low").value;
    el("style-hue-high-label").textContent = el("style-hue-high").value;
    el("style-sat-label").textContent = el("style-sat").value;
    var swLow = el("style-swatch-low");
    var swHigh = el("style-swatch-high");
    if (swLow && window.ExplorerStyle) {
      var s = readStyle();
      var c0 = ExplorerStyle.colorRgba(0.2, 0.05, 0.05, s, { min: 0, max: 1 });
      var c1 = ExplorerStyle.colorRgba(0.95, 0.95, 0.95, s, { min: 0, max: 1 });
      swLow.style.background = "rgb(" + c0[0] + "," + c0[1] + "," + c0[2] + ")";
      swHigh.style.background = "rgb(" + c1[0] + "," + c1[1] + "," + c1[2] + ")";
    }
  }

  function applyStyle() {
    syncLabels();
    var style = readStyle();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(style));
    } catch (e) {}
    ExplorerMap.setStyle(style);
  }

  function restore() {
    if (!el("style-repr")) return;
    var base = ExplorerStyle.defaultStyle();
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (raw) base = Object.assign(base, JSON.parse(raw));
    } catch (e) {}
    el("style-repr").value = base.repr;
    el("style-color-field").value = base.colorField;
    el("style-size-field").value = base.sizeField;
    el("style-base-radius").value = base.baseRadiusM;
    el("style-size-boost").value = base.sizeBoostM;
    el("style-line-width").value = base.lineWidthPx;
    el("style-opacity").value = base.opacity;
    el("style-hue-low").value = base.hueLow;
    el("style-hue-high").value = base.hueHigh;
    el("style-sat").value = base.sat;
    el("style-heatmap").checked = !!base.heatmap;
    applyStyle();
  }

  function bindPopout() {
    var pop = el("style-popout");
    var toggle = el("style-panel-toggle");
    var close = el("style-popout-close");
    toggle.addEventListener("click", function () {
      pop.classList.toggle("open");
    });
    close.addEventListener("click", function () {
      pop.classList.remove("open");
    });
    var ids = [
      "style-repr",
      "style-color-field",
      "style-size-field",
      "style-base-radius",
      "style-size-boost",
      "style-line-width",
      "style-opacity",
      "style-hue-low",
      "style-hue-high",
      "style-sat",
      "style-heatmap",
    ];
    ids.forEach(function (id) {
      var node = el(id);
      if (!node) return;
      node.addEventListener("input", applyStyle);
      node.addEventListener("change", applyStyle);
    });
  }

  if (el("style-panel-toggle")) {
    bindPopout();
    restore();
  }
})();