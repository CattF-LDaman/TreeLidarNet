(function (global) {
  "use strict";

  var DEDUP = 7;
  var flagsById = {};
  var flagsByKey = {};

  function posKey(lon, lat) {
    return round(lon, DEDUP) + "," + round(lat, DEDUP);
  }

  function round(v, d) {
    var p = Math.pow(10, d);
    return Math.round(v * p) / p;
  }

  function indexFlags(list) {
    flagsById = {};
    flagsByKey = {};
    (list || []).forEach(function (f) {
      flagsById[f.flag_id] = f;
      flagsByKey[posKey(f.lon, f.lat)] = f;
    });
  }

  function loadFlags() {
    return fetch("/explorer/api/flags")
      .then(function (r) {
        return r.json();
      })
      .then(function (body) {
        indexFlags(body.flags || []);
        return body.flags || [];
      });
  }

  function flagAt(lon, lat) {
    return flagsByKey[posKey(lon, lat)] || null;
  }

  function isFlagged(lon, lat) {
    return !!flagAt(lon, lat);
  }

  function markFalsePositive(row, opts) {
    opts = opts || {};
    var body = {
      lon: row.lon,
      lat: row.lat,
      bev: row.bev,
      v2: row.v2,
      v3: row.v3,
      reason: opts.reason || "default",
      note: opts.note || "",
      bundle_source: (global.__explorerMeta && global.__explorerMeta.source) || "",
    };
    return fetch("/explorer/api/flags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (r) {
        if (!r.ok) throw new Error("flag failed");
        return r.json();
      })
      .then(function (rec) {
        flagsById[rec.flag_id] = rec;
        flagsByKey[posKey(rec.lon, rec.lat)] = rec;
        if (global.ExplorerMap && ExplorerMap.onFlagsChanged) {
          ExplorerMap.onFlagsChanged();
        }
        if (global.ExplorerDetail && ExplorerDetail.refreshSelection) {
          ExplorerDetail.refreshSelection();
        }
        return rec;
      });
  }

  function unflag(flagId) {
    return fetch("/explorer/api/flags/" + encodeURIComponent(flagId), {
      method: "DELETE",
    }).then(function (r) {
      if (!r.ok) throw new Error("unflag failed");
      delete flagsById[flagId];
      Object.keys(flagsByKey).forEach(function (k) {
        if (flagsByKey[k] && flagsByKey[k].flag_id === flagId) delete flagsByKey[k];
      });
      if (global.ExplorerMap && ExplorerMap.onFlagsChanged) {
        ExplorerMap.onFlagsChanged();
      }
      if (global.ExplorerDetail && ExplorerDetail.refreshSelection) {
        ExplorerDetail.refreshSelection();
      }
    });
  }

  function flagMarkers() {
    return Object.keys(flagsById).map(function (id) {
      var f = flagsById[id];
      return { position: [f.lon, f.lat], flag_id: f.flag_id, reason: f.reason };
    });
  }

  global.ExplorerFlags = {
    load: loadFlags,
    isFlagged: isFlagged,
    flagAt: flagAt,
    markFalsePositive: markFalsePositive,
    unflag: unflag,
    flagMarkers: flagMarkers,
    posKey: posKey,
  };
})(window);