(function () {
  const status = document.getElementById("status");
  const fileInput = document.getElementById("file");

  const map = new maplibregl.Map({
    container: "map",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: [3.7252, 51.0543],
    zoom: 14,
  });

  const overlay = new deck.MapboxOverlay({ interleaved: true, layers: [] });
  map.addControl(overlay);

  function gtColor(d) {
    return d.hasSize ? [52, 211, 153, 175] : [101, 163, 122, 170];
  }

  function updateLayers(payload) {
    const gt = payload.gt_trees || [];
    const pred = payload.predictions || [];
    overlay.setProps({
      layers: [
        new deck.ScatterplotLayer({
          id: "gt",
          data: gt,
          getPosition: (d) => d.position,
          getRadius: (d) => d.radiusMeters || 3,
          radiusUnits: "meters",
          getFillColor: gtColor,
          pickable: true,
        }),
        new deck.ScatterplotLayer({
          id: "pred",
          data: pred,
          getPosition: (d) => d.position,
          getRadius: 3,
          radiusUnits: "meters",
          getFillColor: [56, 189, 248, 200],
          pickable: true,
        }),
      ],
    });
    if (gt.length) {
      map.flyTo({ center: gt[0].position, zoom: 17 });
    }
    const m = payload.metrics;
    status.textContent =
      (payload.chip_id || "chip") +
      " · " +
      gt.length +
      " GT · " +
      pred.length +
      " pred" +
      (m ? " · F1=" + m.f1_at_3m.toFixed(2) : "");
  }

  async function loadFile(file) {
    const buf = await file.arrayBuffer();
    let text;
    if (file.name.endsWith(".gz")) {
      const ds = new DecompressionStream("gzip");
      const stream = new Blob([buf]).stream().pipeThrough(ds);
      text = await new Response(stream).text();
    } else {
      text = new TextDecoder().decode(buf);
    }
    updateLayers(JSON.parse(text));
  }

  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    if (f) loadFile(f).catch((e) => (status.textContent = String(e)));
  });
})();
