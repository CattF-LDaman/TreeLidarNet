(function () {
  "use strict";

  var viewer = { setCrop: function () {} };
  var canvas = document.getElementById("crop-canvas");
  var gpuStatus = document.getElementById("gpu-status");
  var selMeta = document.getElementById("sel-meta");
  var barBev = document.getElementById("bar-bev");
  var barV2 = document.getElementById("bar-v2");
  var barV3 = document.getElementById("bar-v3");
  var btnFlag = document.getElementById("btn-flag-fp");
  var btnUnflag = document.getElementById("btn-unflag");
  var flagReason = document.getElementById("flag-reason");
  var flagNote = document.getElementById("flag-note");
  var flagStatus = document.getElementById("flag-status");
  var selectedRow = null;

  function setBars(row) {
    if (!row) return;
    barBev.style.width = Math.round(100 * row.bev) + "%";
    barV2.style.width = Math.round(100 * row.v2) + "%";
    barV3.style.width = Math.round(100 * row.v3) + "%";
    var flagged = window.ExplorerFlags && ExplorerFlags.isFlagged(row.lon, row.lat);
    var tag = flagged ? " · FLAGGED FP" : "";
    selMeta.textContent =
      "BEV " +
      row.bev.toFixed(3) +
      " · v2 " +
      row.v2.toFixed(3) +
      " · v3 " +
      row.v3.toFixed(3) +
      tag;
    syncFlagButtons(row);
  }

  function syncFlagButtons(row) {
    if (!btnFlag || !btnUnflag) return;
    if (!row) {
      btnFlag.disabled = true;
      btnUnflag.disabled = true;
      return;
    }
    btnFlag.disabled = false;
    var f = ExplorerFlags && ExplorerFlags.flagAt(row.lon, row.lat);
    btnUnflag.disabled = !f;
    if (flagStatus) {
      flagStatus.textContent = f
        ? "Flagged " + f.flag_id + (f.note ? " — " + f.note : "")
        : "Marks this detection as verifier negative (explorer_fp).";
    }
  }

  function loadCrop(row) {
    var url =
      "/api/crop/wgs84?lon=" +
      row.lon +
      "&lat=" +
      row.lat +
      "&score=" +
      row.bev +
      "&source=explorer";
    if (!window.__explorerRescoreOnClick) {
      url +=
        "&verifier_v2=" +
        row.v2 +
        "&verifier_v3=" +
        row.v3 +
        "&use_export_scores=1";
    }
    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error("crop unavailable");
        return r.json();
      })
      .then(function (crop) {
        viewer.setCrop(crop);
        if (crop && crop.verifier_scores && !crop.scores_from_export) {
          var live = Object.assign({}, row, {
            v2: crop.verifier_scores.v2 != null ? crop.verifier_scores.v2 : row.v2,
            v3: crop.verifier_scores.v3 != null ? crop.verifier_scores.v3 : row.v3,
          });
          selectedRow = live;
          setBars(live);
        }
      })
      .catch(function () {
        selMeta.textContent = "Crop unavailable at this location.";
      });
  }

  function onMapClick(lngLat) {
    var idx = ExplorerMap.nearestIndex(lngLat.lng, lngLat.lat, 10);
    var row = ExplorerMap.rowAt(idx);
    if (!row) return;
    selectedRow = row;
    setBars(row);
    loadCrop(row);
  }

  function doFlag() {
    if (!selectedRow || !window.ExplorerFlags) return;
    ExplorerFlags.markFalsePositive(selectedRow, {
      reason: flagReason ? flagReason.value : "default",
      note: flagNote ? flagNote.value : "",
    })
      .then(function () {
        setBars(selectedRow);
      })
      .catch(function (err) {
        console.error(err);
        if (flagStatus) flagStatus.textContent = "Could not save flag.";
      });
  }

  function doUnflag() {
    if (!selectedRow || !window.ExplorerFlags) return;
    var f = ExplorerFlags.flagAt(selectedRow.lon, selectedRow.lat);
    if (!f) return;
    ExplorerFlags.unflag(f.flag_id)
      .then(function () {
        setBars(selectedRow);
      })
      .catch(console.error);
  }

  if (btnFlag) btnFlag.addEventListener("click", doFlag);
  if (btnUnflag) btnUnflag.addEventListener("click", doUnflag);
  window.addEventListener("keydown", function (ev) {
    if (ev.key === "f" || ev.key === "F") {
      if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
      doFlag();
    }
  });

  if (canvas) {
    import("/viewer/crop_viewer_3d.js")
      .then(function (mod) {
        return mod.mountCropViewer(canvas, {
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
        if (gpuStatus) gpuStatus.textContent = "3D viewer failed";
      });
  }

  function refreshSelection() {
    if (selectedRow) setBars(selectedRow);
  }

  window.ExplorerDetail = {
    onMapClick: onMapClick,
    getSelection: function () {
      return selectedRow;
    },
    refreshSelection: refreshSelection,
  };
})();