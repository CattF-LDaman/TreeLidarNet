# TreeLidarNet — Licensing, attribution, redistribution

This is practical guidance for the Ghent pilot — **not legal advice**. When publishing artifacts, read the current source terms linked below.

---

## 1. Source datasets

| Source | Terms pointer | Attribution |
|--------|----------------|-------------|
| **DHMV II LAZ** (EODaS OpenLiDAR) | [Digitaal Vlaanderen — DHMV II brondata](https://www.vlaanderen.be/digitaal-vlaanderen/onze-diensten-en-platformen/basiskaart-vlaanderen-grb/dhmv-ii-brondata); [Vlaanderen open-data voorwaarden](https://overheid.vlaanderen.be/voorwaarden) | Digitaal Vlaanderen / EODaS in UI, README, and papers |
| **Locaties bomen Gent** | [data.stad.gent](https://data.stad.gent/) dataset license | Stad Gent open data credit |
| **GRB** (if used for masks) | [GRB license](https://www.geopunt.be/) | AGIV / Digitaal Vlaanderen as required |
| **Map tiles** (viewer) | [OSM](https://www.openstreetmap.org/copyright), [CARTO](https://carto.com/attributions) | Footer in [VISUALIZATION.md](VISUALIZATION.md) |

EODaS may restrict certain tiles under military air-photography rules — see OpenLiDAR handleiding. Do not redistribute raw LAZ you are not allowed to download.

---

## 2. What we typically publish (intended)

| Artifact | Git / public web | Notes |
|----------|------------------|-------|
| **This repo (code, configs, docs)** | Yes | No LAZ in tree |
| **PyTorch weights (`.pt`)** | Yes | Trained on open data; cite sources in model card |
| **Small inference JSON / GeoJSON** (scores + xy) | Yes | Derived facts; still credit sources |
| **Full `processed/chips/*.npz` archive** | **No by default** | Large; embeds LiDAR-derived geometry — confirm terms before bulk publish |
| **Subsampled point clouds for demo** | Case-by-case | Prefer server-side chip viewer, not bulk LAZ re-export |
| **Trained heatmaps / DSM snippets** | Case-by-case | Low risk at chip scale; avoid redistributing official DSM/DTM rasters without product license |

When in doubt: publish **metrics + code + model weights**; gate **bulk derived point products**.

---

## 3. Model card (minimum)

Each released checkpoint should ship `MODEL_CARD.md` or JSON meta with:

- `gent_trees_parquet_sha256` and `gent_export_fetched_at`  
- `lidar_epoch`: DHMV II 2013–2015  
- Training region (Gent kaartbladen list)  
- Intended use: research / inventory audit — **not** operational forestry or legal surveying  
- Known limits: label offset, post-2015 plantings, hedges, leaf-off acquisition  

---

## 4. Public map ethics (inferred trees)

Showing model predictions on a public map (Luwe playground, HugATree admin) is **low privacy risk** (tree locations only). Frame copy clearly:

- Predictions are **experimental**, not Stad Gent official data.  
- “Discoveries” are candidates for review, not new municipal records.

---

## 5. Checklist before publish

- [ ] Footer credits on any map UI  
- [ ] `split_meta.json` / eval JSON cites Gent hash + DHMV epoch  
- [ ] No `data/raw/laz/` in git or release tarball  
- [ ] README links to this file and [DATA.md](DATA.md) attribution section  

---

## 6. Related documents

| Doc | Contents |
|-----|----------|
| [DATA.md](DATA.md) | Dataset versioning hashes |
| [SPEC.md](SPEC.md) | Risk table |
