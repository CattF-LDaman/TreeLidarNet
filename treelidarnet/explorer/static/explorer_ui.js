(function () {
  "use strict";

  var STORAGE_KEY = "treelidar.explorer.filters";

  function readCfg() {
    return {
      bevMin: parseFloat(document.getElementById("bev-min").value),
      mode: document.getElementById("combine-mode").value,
      tV2: parseFloat(document.getElementById("t-v2").value),
      tV3: parseFloat(document.getElementById("t-v3").value),
      tCombo: parseFloat(document.getElementById("t-combo").value),
      wV2: 0.5,
      wV3: 0.5,
      registerSplitM: 5.0,
      disagreeDelta: 0.15,
      discoveriesOnly: document.getElementById("discoveries-only").checked,
      discoveryMinM: parseFloat(document.getElementById("discovery-min").value),
    };
  }

  function syncLabels() {
    document.getElementById("bev-label").textContent = document.getElementById("bev-min").value;
    document.getElementById("tv2-label").textContent = document.getElementById("t-v2").value;
    document.getElementById("tv3-label").textContent = document.getElementById("t-v3").value;
    document.getElementById("tcombo-label").textContent = document.getElementById("t-combo").value;
    document.getElementById("disc-label").textContent = document.getElementById("discovery-min").value;
    document.getElementById("discovery-wrap").style.display = document.getElementById("discoveries-only")
      .checked
      ? "block"
      : "none";
  }

  var applyTimer = null;

  function apply() {
    syncLabels();
    var cfg = readCfg();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
    } catch (e) {}
    clearTimeout(applyTimer);
    applyTimer = setTimeout(function () {
      ExplorerMap.applyFilter(cfg, function (kept) {
        var meta = window.__explorerMeta || {};
        var src = meta.source ? meta.source.split("/").pop() : "export";
        document.getElementById("visible-count").textContent =
          kept.toLocaleString() +
          " / " +
          (meta.detection_count || kept).toLocaleString() +
          " visible · scores: " +
          src;
      });
    }, 40);
  }

  function bind() {
    var ids = [
      "bev-min",
      "combine-mode",
      "t-v2",
      "t-v3",
      "t-combo",
      "discoveries-only",
      "discovery-min",
      "show-register",
      "hide-flagged",
    ];
    ids.forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      el.addEventListener("input", function () {
        if (id === "show-register") ExplorerMap.setShowRegister(el.checked);
        else if (id === "hide-flagged") ExplorerMap.setHideFlagged(el.checked);
        else apply();
      });
      el.addEventListener("change", apply);
    });
  }

  function restore() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      var cfg = JSON.parse(raw);
      if (cfg.bevMin != null) document.getElementById("bev-min").value = cfg.bevMin;
      if (cfg.mode) document.getElementById("combine-mode").value = cfg.mode;
      if (cfg.tV2 != null) document.getElementById("t-v2").value = cfg.tV2;
      if (cfg.tV3 != null) document.getElementById("t-v3").value = cfg.tV3;
      if (cfg.tCombo != null) document.getElementById("t-combo").value = cfg.tCombo;
      if (cfg.discoveriesOnly) document.getElementById("discoveries-only").checked = true;
      if (cfg.discoveryMinM != null) document.getElementById("discovery-min").value = cfg.discoveryMinM;
    } catch (e) {}
  }

  ExplorerMap.init();
  bind();
  restore();
  document.getElementById("load-status").textContent = "Fetching columnar bundle…";
  ExplorerData.loadBundle(function (msg) {
    document.getElementById("load-status").textContent = msg;
  })
    .then(function (b) {
      window.__explorerMeta = b.meta;
      document.getElementById("load-status").textContent =
        "Loaded " + b.meta.detection_count.toLocaleString() + " detections";
      ExplorerMap.setBundle(b);
      var flagsP = window.ExplorerFlags
        ? ExplorerFlags.load().catch(function () {
            return [];
          })
        : Promise.resolve([]);
      flagsP.then(function () {
        apply();
        if (window.ExplorerDetail && ExplorerDetail.refreshSelection) {
          ExplorerDetail.refreshSelection();
        }
      });
    })
    .catch(function (err) {
      console.error(err);
      document.getElementById("load-status").textContent = err.message || String(err);
    });
})();