/* global ExplorerCombine */
importScripts("/explorer/static/verifier_combine.js");

self.onmessage = function (e) {
  var columns = e.data.columns;
  var cfg = e.data.cfg;
  var gen = e.data.gen;
  var res = ExplorerCombine.filterMask(columns, cfg);
  var mask = res.mask;
  self.postMessage({ gen: gen, mask: mask, kept: res.kept, total: res.total }, [mask.buffer]);
};