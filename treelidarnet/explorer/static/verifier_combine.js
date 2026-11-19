/* Mirrors treelidarnet/explorer/combine.py */
(function (global) {
  function passes(row, cfg) {
    var bev = row.bev;
    var v2 = row.v2;
    var v3 = row.v3;
    var dist = row.dist;
    var hasRgb = row.hasRgb > 0.5;
    if (bev < cfg.bevMin) return false;
    if (cfg.requireRgb && !hasRgb) return false;
    if (cfg.discoveriesOnly && dist <= cfg.discoveryMinM) return false;
    var mode = (cfg.mode || "either_or").toLowerCase();
    if (mode === "disagreement") return Math.abs(v2 - v3) >= cfg.disagreeDelta;
    if (mode === "v2_only") return v2 >= cfg.tV2;
    if (mode === "v3_only") return v3 >= cfg.tV3;
    if (mode === "both_and") return v2 >= cfg.tV2 && v3 >= cfg.tV3;
    if (mode === "either_or") return v2 >= cfg.tV2 || v3 >= cfg.tV3;
    if (mode === "max_score") return Math.max(v2, v3) >= cfg.tCombo;
    if (mode === "min_score") return Math.min(v2, v3) >= cfg.tCombo;
    if (mode === "weighted") return cfg.wV2 * v2 + cfg.wV3 * v3 >= cfg.tCombo;
    if (mode === "register_split") {
      if (dist <= cfg.registerSplitM) return v3 >= cfg.tV3;
      return v2 >= cfg.tV2;
    }
    return v2 >= cfg.tV2 || v3 >= cfg.tV3;
  }

  function filterMask(columns, cfg) {
    var n = columns.bev.length;
    var mask = new Uint8Array(n);
    var kept = 0;
    for (var i = 0; i < n; i++) {
      var ok = passes(
        {
          bev: columns.bev[i],
          v2: columns.v2[i],
          v3: columns.v3[i],
          dist: columns.dist[i],
          hasRgb: columns.hasRgb[i],
        },
        cfg
      );
      mask[i] = ok ? 1 : 0;
      if (ok) kept++;
    }
    return { mask: mask, kept: kept, total: n };
  }

  global.ExplorerCombine = { passes: passes, filterMask: filterMask };
})(typeof window !== "undefined" ? window : globalThis);