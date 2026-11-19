# TreeLidarNet — thought process (design diary)

**Session:** 10–19 November 2026 (build day, full-grid train, anti-carpet pivot, PointNet++ verifier + crop QA + **full city @ 0.80** + **v3 RGB** + **dual verifier QA** + **Flask end-product explorer** + **manual FP flags → verifier retrain** + **weighted export @ 0.45** §47 + **batched verifier export** §49)  
**Repo:** [TreeLidarNet](https://github.com/CattF-LDaman/TreeLidarNet) · references: HugATree, LuweDaman (Deck.GL map UX)  
**Purpose of this file:** A readable narrative for a **project write-up** — what we wanted, how the goal changed, what we built and observed. Technical truth lives in `docs/SPEC.md`, `docs/TASK_GHENT_BROAD.md`, `docs/LABELS.md`, `docs/EVAL.md`.

**Dates in this diary:** Narrative timeline = **10–19 November 2026**. A few **on-disk labels** still say **2025** in the name (e.g. `runs/gent_canopy_v2/archive/pre_full_grid_2025-11-11/`, `…_20251112.json.gz` in export archives) from when those folders were created; configs point at those literal paths. For a polished report, read “11 Nov” in prose as **Nov 2026** unless a quoted path is shown.

---

## One-page story

We set out to detect trees in Ghent from **Flanders DHMV II aerial LiDAR** (2013–2015 point clouds) and compare to Stad Gent’s **open tree register** (~68k rows, ~59k eligible for training). The first plan was essentially **“match the city inventory”** (v1: register positives, negatives everywhere else). That misaligned with reality: the register is **partial** (city-managed trees only), so punishing predictions on unlisted canopy creates false “errors.”

Mid-session we **pivoted** to *“find trees in Ghent broadly”* (canopy v2): keep register locations as **soft positives**, add an **unknown mask** so unlabeled canopy is not penalized, and add **GRB** building/road masks as **hard negatives**. We trained with PyTorch on **50 m LAZ chips**, fine-tuned from v1, ran smoke and **full live city inference** on a Deck.GL map, and pushed code commit `5d48c47`.

After map review, you asked for **more detections without retraining**: lower the **export confidence floor** from 0.35 to **0.20**, add a **confidence slider** on the full city map, and **re-infer** with live polling so rings fill in during the run.

On **16 Nov** the BEV map stays frozen at **`epoch3_hot`**. Work moved to a **PointNet++ crop verifier** (train + QA viewer): rebuild the crop manifest with negatives in val/test, fix how we pick checkpoints, and a local **Crop QA** map (register + amber rings + 3D crop preview) like HugATree’s one-shot tree load. You can see **leafy RGB** in external LAZ viewers; we add **RGB only on the verifier path** (not a 133k BEV retrain) via `data-patch-chips-rgb` on crop-manifest chips.

Overnight the **full 50 m grid** finished (**133k** chips) after an in-place **variable-σ** heatmap patch. On 11 Nov we archived the pre-grid weights and started a long fine-tune. Most grid chips have **no register trees**; we **oversampled** the ~10.7k that do. Epoch 1 of that pass landed at loss **0.000394** with only a slight drop in smoothed `loss/pos`. **Per-tree peak loss** (weight **0.01**) is already in the config; a watcher starts it when epoch 2 finishes, so each missed stake can move the weights without a hard “every tree must be detected” rule. That term does **not** forbid discoveries. **Audit F1** is the score we read after infer; training still picks `best.pt` by train loss. Later, once detections are trustworthy, a **second network** scores each candidate on a small LiDAR crop: tree or not, and only a coarse structure class. It **trains on every eligible register tree**, including stakes the detector missed. The detector only chooses where to run it. The register itself stays the rich species table.

---

## Mini glossary

| Term | Meaning |
|------|---------|
| **DHMV II** | Vlaanderen digital height model / LiDAR campaign used here (LAZ tiles). |
| **LAZ** | Compressed LiDAR point files per 500 m tile. |
| **Chip** | 50 m × 50 m patch: points, BEV features, heatmap target, masks (~235 KB `.npz`). |
| **BEV** | Bird’s-eye view grid (0.25 m cells) fed to a small CNN. |
| **Register / inventory** | Stad Gent `locaties-bomen-gent` — stake points, not every tree in LiDAR. |
| **Heatmap head** | Model outputs a 2D map; **peaks** above a score threshold become tree candidates. |
| **Discovery** | Prediction farther than **5 m** from every register tree. Allowed. Audit F1 does not count it as a false positive; strict F1 does. |
| **Strict vs audit F1** | Both computed at **infer / eval**, shown on the map. Audit is the primary comparison (discoveries excluded). Strict counts every unmatched ring. **Neither** is used while training. |
| **Unknown mask** | Pixels far from any register point → no negative training loss (canopy may exist without a label). |
| **GRB hard negative** | Official building footprints + roads (buffered) → extra penalty for predicting trees on built infrastructure. |
| **Fine-tune** | Start from v1 `best.pt` weights; adjust for v2 loss/masks rather than train from scratch. |
| **Confidence score** | Model output 0–1 on each peak; **export floor** is the minimum kept at infer; map **slider** can hide weaker rings without re-infer. |
| **Variable σ** | Training heatmap Gaussian width from register **`stamomtrek`** (1.0–2.5 m); addresses stake-sized targets vs wide crowns on large trees. |
| **NMS** | After peak pick: drop weaker detections within ~2 m — fixes duplicates, not “no peak at all.” |
| **CHM** | Canopy height model: DSM − DTM. **CHM peaks** are hilltops on that height map (non-learned baseline). |
| **Validation F1** | The same audit (and strict) scores on a **held-out** area. Not inside this train loop. Meant as a checkpoint check before the next full-city infer. |
| **Per-tree peak loss** | Implemented, weight **0.01**. Starts when epoch 2 ends: focal penalty on the strongest prediction within **2 m** of each register peak. |
| **Candidate network** | Second-stage **PointNet++** crop verifier. **Train** on a LiDAR crop of **every** eligible register tree, including stakes the detector missed. **Run** only on detections (§37). |
| **Crop QA** | Local `serve-crop-qa` + Deck map + 512-pt 3D preview. Register loaded once (HugATree-style), not per-viewport. |
| **Verifier checkpoint** | `best.pt` chosen by **val F1** (both classes in val), not raw val BCE alone (§40). |
| **Explorer** | Flask **end-product QA** map: full dual-scored bundle, combine filters, 3D crop dock (§45). |
| **`explorer_fp`** | Manual false-positive flag at a detection peak → **`label=0`** crop row for verifier retrain (§46). |
| **Verifier combo** | Dual export keep: **`0.5·v2 + 0.5·v3 ≥ combo-t`** (default **0.45**); stored as **`verifier_score`** on shipped gz (§47). |

---

## Timeline (dense)

| Window | Your focus | Outcome |
|--------|------------|---------|
| **01:05–01:10** | Human spec; PyTorch; Deck.GL like HugATree/Luwe; gap review; **uv** for env | `docs/SPEC.md` + gaps plan; uv-first project scaffold |
| **01:27–01:41** | TensorBoard (not custom viz); real Gent import; crash; **what are chips?** vs 68k trees; LAZ download; disk/time | Chips = spatial batches, not 1 chip per tree; manifest + download pipeline |
| **01:42–01:59** | Architecture for DL beginner; **register is incomplete**; Flanders-wide reuse; **discoveries**; OSM later? | Docs on partial labels; v1 still register-centric |
| **02:02–02:42** | Chips done → train; epochs/overfitting fear; early infer; **checkpoint cadence** | v1 training `gent_real_v1`; more frequent `latest.pt` |
| **10:51–11:02** | Full Deck.GL demo; live map; **UX**: zoom, heatmap lag, legend confusion | Scatter rings (HugATree-style), `ghent_live.html`, polling exports |
| **11:47–11:49** | Satellite + softer negatives intuition; **private GitHub** | Repo created; code pushed, data ignored |
| **11:54–12:53** | **Pivot: find trees broadly**; rebuild; **why full grid?** | `TASK_GHENT_BROAD.md`, unknown mask in chips/loss; full grid **deferred** (RAM) |
| **14:22–15:10** | Start v2 train; **GRB now**; DSM 1m?; **restart fresh** after GRB patch | GRB download (buildings ~1.6k, roads ~38k features); chips repatched; train from v1 init |
| **15:34–15:50** | Trees near roads/houses; read **loss/LR/pos**; jumpy TB; GPU; **faster DataLoader** at ~step 500 | `num_workers`, `pin_memory`; resume semantics clarified |
| **16:05–16:36** | Epoch 1–2; smoke infer; **missing vs extra trees**; full grid while training? **QGIS kill** | Smoke map; epoch loss ~3.3e-4; don’t build full grid parallel on laptop |
| **16:39–17:02** | LAZ tile lookup (WGS84→31370); epoch 3; **full live infer**; **0 chips but green trees** | Live infer; flush bug (batch 8 vs 150); viewer guard; training stopped |
| **17:24** | **Map QA:** large trees/clusters silent; small/individual register trees sharp; off-register hits | Qualitative v2 story matches design |
| **17:49** | Repo up to date? | Commit `5d48c47` canopy v2 + fixes pushed |
| **17:51–17:53** | **THOUGHT_PROCESS.MD** for write-up; easy read, DL-literate audience | Design diary at repo root |
| **18:02–18:06** | Full-grid chips: **append** detections? Live map **threshold** slider? | Merge via one infer + NMS; slider on full map (not live first) |
| **18:22–18:25** | More **training** vs tweak knobs? **Re-infer** after threshold? σ / retrain? | Resume training = refinement; σ/chips need fine-tune; τ/slider = infer-only |
| **18:28–18:33** | Lower peak export to **0.20** (plain language); **re-infer only**; **live map** on that run | CLI `--score-threshold`; `ghent.html` slider; live infer restarted with `--live` |
| **22:xx** | Full **50 m grid** chip build started (~139k origins); map QA → large crowns weak under fixed σ | Grid running in background; paused ~40k steps (~50k npz) to ship **variable σ** without LAZ rebuild |
| **22:xx** | **Patch heatmaps in place**; resume grid with `--skip-existing` | `data-patch-chips-heatmap` (GRB refresh on); docs + `tests/test_heatmap_sigma.py` |
| **22:30–22:50** | Chips = 2D targets not 3D blobs; export does **not** need σ; what is NMS; clusters often have **zero** rings | §14; NMS never runs if there is no peak |
| **22:49** | Push variable σ | `952033d` |
| **11 Nov ~02:00** | Grid **done** | **133,000** index rows; **10,686** with `tree_count > 0` (~8%) |
| **11 Nov 10:34** | More training; **back up** old weights | Archive `pre_full_grid_2025-11-11/`; train on full index, fresh optimizer, init archived `best.pt` |
| **11:20–11:31** | Is the model learning from empty chips? CHM? Unsupervised features? | Empty chips: almost no tree target (unknown mask); CHM/unsupervised/bigger net **deferred** |
| **11:31** | Compute should go to **identifying trees** | `tree_chip_weight: 15` → ~57% tree draws; resume step ~9k. Register chips repeat ~7× per epoch |
| **11:57** | GIF of last full infer, confidence **20%↔40%** | `exports/inference/gent_confidence_20_40_loop.gif` (46k → 981 preds) |
| **12:59–14:30** | Little downward `loss/pos`? Bigger net? | Slight smooth decline is normal; do **not** widen the CNN yet |
| **15:17** | Epoch 1 of this pass **finished** | Loss **0.000394**; epoch 2 started; **wait** before infer |
| **16:05–16:14** | Diagnose training; validation F1; force every register tree? | Grid-mean loss dilutes misses. Next: **per-tree peak loss** after epochs 2–3. Val F1 is a later checkpoint check, not inside this loop |
| **16:23–16:25** | Later: detections only, then a network for type / “is it a tree”? | **Second stage** on LiDAR crops of **every** eligible register tree, including ones the detector missed. Species stays coarse. Register remains the attribute table |
| **16:30–18:12** | Epoch 3 should already use the new loss; anything else before cutover? | Per-tree loss in config. Watcher snapshots epoch 2, then restarts. Weight **0.01** (not 1). Pushed `70376a9` |
| **18:40–18:45** | Will discoveries survive? Is **5 m** a lot? Do we even use audit or strict F1? | Discoveries stay allowed. **5 m** is only the audit bucket (match is **3 m**, train/NMS **2 m**). Audit F1 is the infer score we look at; strict sits beside it; training ignores both |
| **19:07–19:14** | Epoch 3 starts, then **NaN** at step **76,511** | `loss/tree` was falling (0.47 → 0.23). A saturated fp16 score became exactly 1, and `-log(1 - pred)` was NaN |
| **19:58** | Fix and restart epoch 3 from the epoch-2 weights | Loss in **fp32**, grad clip **5.0**. Do not continue the 1,000 heated steps |
| **20:11–20:20** | **12 m** unknown band is too wide for front gardens | Paused at step **76,992** (Adam kept). Patched **10,686** chips to **5 m** in 5.5 min. Resumed |
| **22:00–22:05** | How are the scalars? Are we doing okay? | Halfway through the restarted epoch. `loss/tree` **0.075 → 0.044**. `loss/pos` crept **0.69 → 0.79**. Keep going; no change mid-pass |
| **22:06–22:12** | How much more training until this model is its best? Wider U-Net? Are the points themselves used? Should shape have been in the model? | Finish **this epoch**. Do not widen the U-Net. Points are averaged into cells; height above ground is stored and unused. Shape comes later as height channels, then the crop model |
| **22:19** | Keep the LiDAR points of each detection for **Ghentree** | Same crop the second model sees. Later: a static isometric image of those points, or a small interactive 3D view. Not a stand-in made of circles |
| **22:24** | How long will the infer after this epoch take? | About **0.37 s/chip**. Full **133k** is **13–14 h**. The **10,686** register chips are about **1 h**. Do not infer while training holds the GPU |
| **22:27** | Infer should fill the map from the **center of Ghent** outward | `infer-gent` sorts chips by distance from the middle of the grid. The finished detections do not change. `--limit` now keeps the center |
| **22:34** | What is `loss/train_step`? | The loss of one batch: grid focal loss plus **0.01 × `loss/tree`**. That is what the printed epoch loss averages. Near **0.0015** on this pass |
| **22:36–22:37** | The detector writes a 200×200 heatmap and we pick the highest peaks? Is **0.25 m** fair for one tree each? | Local maxima at **≥ 0.20** inside **2 m**, then NMS. The cell places a tree to about **12 cm**. Two trunks inside **2 m** still become one detection |
| **22:51–22:55** | Is a missed tree inside a chip punished? Should the ignore ring change? | The per-tree term punishes a stake it can see. The **8–15 m** ignore ring hides about **30%** of stakes, and tiles do not overlap. Clear it after this epoch, then one more epoch |
| **22:58** | Register chips before empty ones, still outward | Sort from the center first, then put `tree_count > 0` in front. Both groups keep that order. Training order is unchanged |
| **23:10–23:15** | Can infer go faster without changing detections? | Yes. The Python point loop was the **0.37 s/chip**. Same sums via `index_add_`, four workers, and infer skips the training arrays. A 24-chip grid build is now **0.005 s/chip**. Scores and peaks stay the same |
| **12 Nov 00:00–00:15** | Epoch 3 done. Live center looks carpeted. Full infer should be live, with a slider | Training stopped. `best.pt` is epoch 3, loss **0.0021**. 400 central chips → **20,398** rings. Usable near **0.85**. Full live infer running. Amber rings draw over the green |
| **12 Nov 00:17** | The model has to stop inventing trees | Slider hides the carpet. Next epoch penalizes unknown peaks. Discoveries of real unlisted crowns stay allowed |
| **12 Nov 00:21** | High-confidence rings look like trees. Keep them | The 50 m grid is the **0.50–0.85** corner lattice. From **0.90** the frame is gone. The penalty must not use distance-to-stake, and trees already at **≥ 0.90** have to survive |
| **12 Nov 00:36** | East–west streaks have no LiDAR. Fetch the other flight line tomorrow | **25,979** chips have zero points, **1,841** of them with register trees. The downloader saved the strip under the tile center. The strip that fills the northern edge was never requested |
| **12 Nov 01:06** | Second model trains on every register tree | Crops come from **all** eligible stakes, including trees this detector never rang. Detections are where it **runs**, not where it **learns** |
| **12 Nov 01:24** | Crop classifier is PointNet, then a small PointNet++ | The city detector stays the BEV U-Net. The crop is a vertical stack: column, then crown. PointNet first. Two-scale PointNet++ if poles and shrubs still pass |
| **16 Nov** | **PointNet++ verifier immediately** | Skip staging plain PointNet; two-scale PointNet++ is the first implementation (§37, [docs/SPEC.md](docs/SPEC.md) §6.5) |
| **16 Nov** | **Crop QA map** pop-in; 3D panel broken | Full register load (`/api/register/all`); Three.js **import map** + WebGPU `renderAsync` fallback |
| **16 Nov** | **Are crop losses right?** | Plain BCE is fine math; **val had no negatives**, unweighted BCE, **zero placeholders** — fixed before trusting `best.pt` (§40) |
| **16 Nov** | **`gent_crop_verifier_v2` train** | Rebuilt `crops_index`; balanced loss; **`best.pt` by val F1**; TensorBoard on **6006** (start `make tensorboard` yourself) |
| **16 Nov PM** | **Verifier-only RGB (§41–§43)** | Manifest walked; **register-first** re-patch (**2,166** tiles); full grid RGB **paused**; then **`v3_rgb`** train — BEV unchanged |
| **16–17 Nov** | **v2 vs v3 compare on crop QA** | Dual checkpoints on `serve-crop-qa`; click + map halos via `filter-export --live` with **`--verifier-v2`** + **`--verifier-v3`** (§44) |
| **17 Nov AM** | **Full-grid RGB finished** | **`--all-chips --only-missing-rgb`**: **79,460** writes / **83,290** queued (`patch_chip_rgb_full_city.log`) |
| **17 Nov AM** | **Full dual verifier score** | **`filter-export-verifier-full`**: **530,258** preds @ **BEV ≥ 0.85**, v2+v3 compare + live poll — not BEV `infer-gent` |
| **17 Nov AM** | **Almost started BEV re-infer by mistake** | `make infer-gent-full` killed **~16k/133k**; **`gent_city_gent_canopy_v2.json.gz` unchanged** (export only at end of infer) |
| **17–19 Nov** | **§45 Flask explorer shipped** | `data-build-explorer-bundle`; **`make explorer-serve`** @ **8765**; **~402k** dual-scored dets in columnar `.bin`; Style pop-out; combine presets; 3D Z-up orbit |
| **19 Nov** | **§46 Explorer FP flags** | Dock **Mark false positive** / **`f`** → `explorer_qa_flags.parquet`; merge **`source=explorer_fp`** into `crops_index`; map layer + hide flagged |
| **19 Nov (ask)** | **Building FPs in product** | GRB ≠ peak location; **export keep = v3 @ 0.5** (not explorer AND/OR); **~30–80 diverse flags** + retrain + re-export; optional later **`explorer_fp` loss upweight** |
| **19 Nov PM** | **§47 Weighted export gate** | **`filter-export` dual keep = 0.5·v2 + 0.5·v3 ≥ 0.45** (slightly looser than v3-only @ 0.5); explorer can filter tighter; re-export + bundle to refresh on-disk gz |
| **19 Nov PM (ask)** | **v2 + v3 is the product** | LiDAR-only v2 kinder on discoveries; RGB v3 stronger on register-like crowns; **combine policy** is the spine — not “pick one verifier” |
| **19 Nov PM (ask)** | **RGB in v3** | **Per-point** DHMV color on the ~512 crop — **not** ortho/satellite tiles; MapLibre basemap is human QA only |
| **19 Nov PM (ask)** | **Explorer score display** | Bundle **bev/v2/v3 as-is** (0–1 sigmoid / heatmap); combine sliders use raw values; **map hue** min–max over **visible** detections only |
| **19 Nov PM** | **§48 Post-re-export FP loop** | Frozen BEV; full weighted re-export first; later flags → incremental `_fp` fine-tune + **filter-export** on same gz |
| **19 Nov night** | **§49 Batched verifier export** | Chip-grouped crops + GPU batch forward; **`--infer-batch-size 64`**; resume from `live/verified_*.json` (stop job before restart) |

---

## Phased narrative

### 1. Specification and build (early morning)

**Context.** You asked for a comprehensive spec: neural tree detection from Vlaanderen LiDAR plus Gent’s bomen dataset, PyTorch, Deck.GL visualization patterned on your tree-map projects.

**Your questions.** Gap analysis? Then **Build** with **uv** managing the environment.

**Outcomes.** SPEC, DATA, EVAL, LABELS, DEV docs; `treelidarnet` CLI; chip pipeline; small BEV detector (heatmap + focal loss).

**For write-up:** Motivation = operational tree inventory is incomplete; LiDAR sees canopy citywide.

---

### 2. Data: LAZ, chips, and the 68k vs 10k confusion

**Context.** Long-running download and chip build; you wanted visible progress and disk/time estimates.

**Your questions.** What are **tree chips**? Why ~10k chips if the dataset has **68k** trees?

**Outcomes.** ~10.7k **chips** are 50 m tiles that **contain** at least one eligible register tree (many trees per chip). ~59k **eligible** points are label targets inside those chips. Full Gent bbox grid was sketched but not required for v1.

**For write-up:** Training unit = geographic patch, not one row per tree.

---

### 3. v1 training and label alignment

**Context.** Real training with TensorBoard; system crash; checkpoint frustration.

**Your concerns.** Only **city-maintained** trees are labeled. Will long training teach the model to **suppress** unlisted trees? You want **other municipalities** and **discoveries** in Ghent; OSM might help later.

**Outcomes.** v1 (`gent_real_v1`) learned **register matching** well enough for a first city demo. Documentation stressed partial inventory and dual metrics (strict vs audit). Overtraining concern noted — addressed properly only after pivot.

**Decision:** v1 is a baseline and initialization for v2, not the final product definition.

---

### 4. Inference UX and first city demo (morning)

**Context.** Full-Ghent infer and a **live** map that fills in as chips complete.

**Your feedback.** Heatmap layer laggy at zoom; hard to tell **register vs prediction**; wanted HugATree-smooth scatter.

**Outcomes.** `ghent_live.html` polls `exports/inference/live/`; green filled circles = register, amber rings = model; dropped live heatmap overlay for performance.

**Decision:** Demo UX = **scatter + legend**, not raster heatmap on the basemap.

---

### 5. Pivot — “Find trees in Ghent broadly” (midday)

**Context.** Your phrase: pivot to broad canopy finding, not census replacement.

**Your questions.** Why **full-grid** chips if we already have register-skewed chips?

**Outcomes.** [docs/TASK_GHENT_BROAD.md](docs/TASK_GHENT_BROAD.md): primary product = LiDAR **candidates**; register = weak supervision + benchmark. **Unknown mask** (default 12 m band): no negative loss far from register points. Config `gent_canopy_v2.yaml`; eval emphasizes **audit F1** and **discovery count**.

**Decision:** Task definition changed; v1 weights reused via **fine-tune**, not a new architecture.

**For write-up:** This is the main methodological pivot — from supervised matching to partial-label canopy search.

---

### 6. GRB hard negatives and training restart (afternoon)

**Context.** You stopped training to add hard negatives **immediately** (GRB: official buildings + roads).

**Your questions.** How are hard negatives chosen? **DSM 1 m** included / would it help? **Resume** after GRB patch or **full restart**? Are hard negatives actually in the run? Trees **next to roads/houses** wrongly punished?

**Outcomes.** `download_grb`, rasterize into chips; peak guard so GRB does not mask register peaks. First patch bug (CRS) → all-zero hard mask fixed. **Fresh** `gent_canopy_v2` training from `gent_real_v1/best.pt` after repatching ~10.7k chips — not resume from pre-GRB optimizer state.

**Decision:** Hard negatives on **infrastructure only** (buffered), soft unknown band for street trees; DSM deferred (optional future channel).

---

### 7. Reading training (learning curve)

**Context.** You’re new to TensorBoard scalars; worried about jumpy **`loss/pos`**.

**Your questions.** What is **LR**? What does **`loss/pos` ≈ 1.5** mean? Is jumpiness bad? GPU usage? Speed up without breaking training?

**Outcomes.** `loss/epoch` ~0.00033 stable; `loss/pos` noisy (batch of 4, sparse positive pixels, per-batch mean). DataLoader workers + pin_memory mid-run. ~2672 steps/epoch (10 686 chips ÷ batch 4). Epoch 1–3 completed in session.

**For write-up:** Report epoch loss + qualitative map; don’t over-interpret step-level `loss/pos`.

---

### 8. Smoke test and epoch 2 check (late afternoon)

**Context.** Epoch 1 `best.pt`; smoke infer 200 chips; map viewer.

**Your observations.** Some **small trees not in register** detected; many **register trees missing** in smoke; many unlisted trees visible — expected at early epoch + partial coverage.

**Outcomes.** `ghent_smoke.html`; metrics on 200 chips not representative of full city. Advised against **full-grid chip build** while training on memory-tight laptop; killed QGIS/untwine indexing LAZ.

**Decision:** Defer `data-build-gent-grid` until machine headroom; keep register-skewed set for train/infer.

---

### 9. Full live infer and map confusion (evening)

**Context.** Stop training at epoch 3; **full live infer** like v1 (~10.7k chips).

**Your confusion.** UI said **0 chips** but **green trees everywhere**; rings appeared then vanished.

**Outcomes.** Green = full register loaded at start (`gt.json.gz`), not infer progress. Counter stuck until live flush; bug: flush only when `done % 150 == 0` but `done` steps by **batch 8** — first flush at 152/304/…; fix uses **boundary crossing**; status updates every 50 chips; viewer won’t wipe rings on empty JSON. Training saved `latest.pt` at epoch 3, step ~8843.

---

### 10. Qualitative findings on the live map (your monitoring)

**Observation (your words, summarized).** Large trees and dense clusters often get **no** model ring; **small / individual** register trees often match **very accurately**; many detections **outside** the register (discoveries).

**Interpretation (for write-up).** The head is peak-based (local max on heatmap, τ≈0.35): isolated street canopies → sharp peaks; wide crowns → diffuse response below threshold; register point at stem vs crown centre hurts large trees; clusters need multiple peaks — NMS and training still register-shaped. Green disk size = `stamomtrek`, amber = fixed 3 m — visual “miss” on big trees is exaggerated. v2 **allows** off-register fires by design.

**For write-up (limitations):** Strong on stake-like street trees and discovery layer; weak on large-crown and multi-tree ambiguity without more training, threshold tuning, or instance-style targets.

---

### 11. Post-demo: coverage, confidence, and maps (late evening)

**Context.** Full live infer and map monitoring led to questions about scaling coverage, tuning without retraining, and repo hygiene.

**Your questions.** Sync repo to GitHub? Document the day in **THOUGHT_PROCESS**? Will **full-grid** chips **append** to existing city detections? **Slider** on maps for confidence? More **epochs** or change **threshold / σ**? Must we **re-infer**? Explicit ask: lower export confidence so the **slider can show weaker trees** (you preferred natural language over symbols like τ).

**Outcomes.**

- **Git:** `5d48c47` pushed; later map/threshold code may be local until next commit.
- **THOUGHT_PROCESS.MD:** this file — timeline for your report.
- **Full grid:** new chips land in the same `chips/` folder; city list = **one infer** over all `.npz` with merge NMS — not a manual “append” API.
- **Training vs tuning:** more epochs (resume toward 7–10) helps sharpness on **existing** chips; **σ / unknown band** need chip rebuild + fine-tune; **confidence floor** is **infer-only**.
- **Re-infer rule:** map slider filters the export file only; new floor or checkpoint → run `infer-gent` again.
- **Implementation:** default export **min confidence 0.20** (`DEFAULT_PEAK_SCORE_THRESHOLD`); `peak_extract_min_score` in city JSON; full map slider default **0.35** view; **`--score-threshold 0.2`** on CLI.
- **Ops:** first non-live re-infer restarted for **`--live`** so `ghent_live.html` updates; prior partial run at ~600 chips discarded.

**For write-up:** Separating **model training**, **export policy** (how many peaks we keep), and **exploration UI** (slider) is a deliberate product choice for iterative QA.

---

### 12. Full-grid chips and variable heatmap σ (late evening)

**Context.** After lowering export confidence and re-inferring, qualitative map QA still showed **large crowns and clusters** under-detected while street stakes matched well — partly peak extraction (τ), partly **fixed 1.0 m** training Gaussians at register points (crown centre offset from stake).

**Your direction.** Start the **full Gent 50 m grid** for citywide coverage, but adopt **σ from `stamomtrek`** without throwing away chips already built from LAZ.

**Outcomes.**

- **`heatmap_sigma_m`:** `σ = clamp(0.9 + 0.015 × stamomtrek, 1.0, 2.5)` m in `treelidarnet/data/chips.py`; new chips from `data-build-chips` / `data-build-gent-grid`.
- **`data-patch-chips-heatmap`:** Rewrites `heatmap` (+ optional GRB refresh) in existing `.npz` via atomic `{stem}.tmp.npz` → replace — **no LAZ pass**.
- **Grid pause:** Full grid job stopped ~**40k / ~139k** loop steps (~**50k** `.npz` on disk) so patch could run without file races.
- **Resume:** `data-build-gent-grid --laz-dir data/raw/laz --skip-existing` after patch finishes; new cells get variable σ at build time.

**Decision:** **Patch-then-resume** beats full grid rebuild; **fine-tune** (or more epochs) needed after heatmap change — export threshold alone does not fix training target scale.

**For write-up:** Link map weakness on large trees to **label geometry** (point target + fixed σ) and separate **infer policy** (0.20 floor) from **supervision shape** (girth-scaled σ).

---

### 13. Levers for better detection (plain-language map)

**Context.** Late in the day you asked whether **variable σ** would really help, and how it relates to everything else we had already tried (export **0.20**, GRB, unknown mask, full grid). This section is the **beginner-friendly cheat sheet** for the write-up — same story as the code, without re-reading EVAL/SPEC.

**One picture.** Detection has three layers:

1. **What we teach** — heatmap blobs, masks (unknown, GRB, edges).
2. **What the model learns** — CNN on BEV LiDAR, focal loss on heatmaps; more chips and epochs.
3. **What we keep after infer** — peak picking, NMS, score floor, map slider.

Changing layer 3 **does not retrain** the network; changing layer 1 **requires retrain** to bite.

#### The model in one sentence

Each **chip** is a 50 m window of LiDAR seen from above. The network outputs a **heatmap** (“tree-ish here”); we turn **bright local bumps** into candidate trees ([EVAL.md](docs/EVAL.md)).

#### Variable σ (what we are patching now)

Training targets are **soft Gaussians** at register `(x, y)`, not single pixels.

| Before | After |
|--------|--------|
| Every tree → ~**1 m** wide blob | **Large `stamomtrek`** → wider blob (up to **2.5 m**); small/unknown → still ~1 m |

**Why we bothered:** Map QA showed **large crowns** often weak while **street stakes** matched well. LiDAR “tree-ness” spreads across the crown; a 1 m dot at the **stake** under-supervises the crown footprint. Wider blobs are a cheap alignment with girth — still centred on the register, **not** the crown centroid.

**Expectation:** **Moderate** gain on big register trees after **fine-tune**; not a fix for clusters or unlisted trees alone.

#### Other levers (what helps what)

| Lever | Teaches the model? | Main effect |
|-------|-------------------|-------------|
| **Variable σ** | Yes (labels) | Bigger positive target for bigger trees |
| **Full grid chips** | Yes (data coverage) | Train/infer on whole Gent ∩ LAZ, not only register neighbourhoods |
| **More epochs / fine-tune** | Yes | Sharper, stabler heatmaps on existing pipeline (~3 epochs was early) |
| **Export 0.20 + slider** | **No** | Keep/show weaker peaks the model **already** made |
| **GRB hard negative** | Yes (mask) | Penalize trees on roofs/roads (already in chips) |
| **Unknown mask (v2)** | Yes (mask) | Don’t punish unlisted canopy; discoveries allowed |
| **Peak extract / NMS / top-K** | No (infer rules) | How bumps become a **list** of trees; cluster trade-offs |
| **Offset head** (not built) | Yes (architecture) | Predict Δx,Δy from peak — classic fix for stake vs crown |
| **CHM / DSM channel** (deferred) | Yes (inputs) | Dense height cue; extra download + chip work |
| **Stratified metrics** | N/A (evaluation) | Split by `stamomtrek`, park vs street — see if σ moved the right bucket |

**Analogies we used in chat** (for the report narrative):

- **Lower τ / slider** — turning down a “minimum score to pass” filter: you **see** more students, you didn’t teach them better.
- **More epochs** — more practice with the same answer key.
- **Full grid** — practicing the whole city map, not only streets that already have inventory signs.
- **σ** — the answer key blobs got **wider** for big trees; training still required to learn from the new key.

#### What variable σ does *not* fix (honest limits)

- **Stake vs crown centre** — target is still the inventory point; wide oak can be “right” off-centre in LiDAR.
- **Clusters** — one Gaussian per `gis_id`; overlapping crowns still fight **one peak / NMS**.
- **Diffuse heatmaps** — training may encourage broader bumps; **peak extract** can still miss flat responses unless infer rules change.
- **Discoveries** — unlisted trees depend on **unknown mask + coverage**, not girth labels.

#### Bundled plan vs science fair knobs

**Tonight’s agreed bundle** (patch → grid → train → infer) hits the best **first bundle**: fix labels (σ), widen coverage (grid), then **re-learn** (fine-tune). **0.20 export** was the right **QA** move earlier; it does not substitute for that bundle.

**For write-up:** Frame improvements as **three layers** (teach / learn / keep). Session priority = layer 1+2 for large-tree complaints; layer 3 already partially opened for exploration.

---

### 14. Consolidated mental model (chips, export, NMS, silent clusters)

**Context.** After σ landed, you checked understanding: what’s **inside** a chip, whether **export** must change, what **NMS** does, and why **clusters** often show **no** ring at all — not merely duplicate rings.

#### What a chip actually stores

A chip is **not** a 3D blob collection for the network. **`xyz` / `feat`** are 3D LiDAR; everything else is a **2D bird’s-eye grid** (200×200, 0.25 m):

| Array | Meaning |
|-------|---------|
| `heatmap` | **Training target** — register Gaussians (now **variable σ** per `stamomtrek`) |
| `ignore` | Was an 8–15 m edge annulus. Now all zeros (§30). Stored chips still have the ring until the patch |
| `unknown` | v2 — no negative loss far from register |
| `hard_negative` | **GRB** roads/buildings rasterized on the same grid (not a separate 3D building model) |

**Fine-tune story:** New σ changes **targets** only. Warm-start from `gent_canopy_v2/best.pt` on patched + new chips is the graceful path; it still requires **retrain**, not infer-only.

#### Export and σ

**Infer / export do not implement σ.** σ lives in chip **labels**. Pipeline: LiDAR → model heatmap → **peak extract** + NMS → JSON. Re-run **`infer-gent`** after a new checkpoint; same `--score-threshold` unless you deliberately tune postprocess.

#### NMS (non-maximum suppression)

After local maxima above τ are found, **NMS** sorts by score and **drops weaker peaks within 2 m** of a kept peak (`postprocess.py`). It fixes **too many** detections on one tree — “several shouts, keep the loudest.”

#### Silent clusters (your observation)

Map QA: **large groups often have zero detections**, not just merged duplicates. That usually means **no peak cleared the pipeline**, so **NMS never ran**.

| Symptom | Likely cause |
|---------|----------------|
| **No ring** in a cluster | Diffuse / weak heatmap, or no local max above τ — **model + targets + peak extract** |
| **Many rings** on one tree | Peak extract + **NMS** |
| **One ring** for a whole group | One broad bump; NMS collapses neighbours |

Overlapping training Gaussians, stake vs crown, ~3 epochs, and register-skewed chips favour **sharp street spikes** more than **N peaks in 10 m**. Lower τ helps only where a weak peak **exists**; it does not lift a flat plateau. Variable σ targets **big single register trees** more than **instance splitting in rows**; clusters may need more training, full grid, and later instance-style ideas — not “relax NMS” as the first fix.

**For write-up:** Separate **missing peaks** (heatmap/training) from **duplicate peaks** (NMS). Clusters are the former on your map.

---

### 15. Grid finished; what empty chips teach (11 Nov morning)

**Context.** `data-build-gent-grid --skip-existing` finished overnight: **133,000** chips in `chips_index_full.parquet`, **10,686** with `tree_count > 0` (~8%). A light sample of chips (350 files) showed sane heatmaps and GRB masks; tiny files were empty LAZ cells, not corruption. Pre-grid `best.pt` / `latest.pt` copied to `runs/gent_canopy_v2/archive/pre_full_grid_2025-11-11/`. Training started on the full index with a **fresh optimizer**, weights loaded from that archive (`init_checkpoint`). **Not** `--resume` of the old 10.7k run.

**Your questions.** With so many chips that contain no register trees, is the model learning anything there? What are **CHM peaks**? Would adding them as features help **now**? Does the full grid mean **more than half** the time is spent on chips with little tree information?

**Answers we settled.**

- A no-tree chip has a **zero heatmap**. The unknown mask is **all ones**, so soft negatives are off. The only real penalty is **GRB** where the model fires on roads or roofs. Those chips do **not** teach “tree here.”
- **CHM** = DSM − DTM (height above ground). **CHM peaks** are local hilltops ≥ ~3 m — the non-learned baseline in `treelidarnet/baseline/chm.py`. Not in the current 4-channel BEV.
- Adding CHM or a max-height channel **during this run** would change `in_channels` and force a rebuild. Finish this experiment first.
- Uniform sampling really is ~**92%** steps with no register target. You do not get 10× more positive examples; you get ~10× more steps for the same ~11k tree chips.

**For write-up:** Full grid is the **infer** set. It is a weak teacher unless sampling or the loss compensates.

---

### 16. Tree-weighted sampling (11 Nov, late morning)

**Your call.** Identifying trees has always been the point; stop spending most of the GPU on empty chips.

**What we did.** Stopped the uniform run around step **9,000**. Added `sampler.tree_chip_weight: 15` (`WeightedRandomSampler`, replacement). Same **33,250** steps per epoch (133k ÷ batch 4), but ~**57%** of draws are register chips. Each of the 10,686 positive chips is seen about **7 times** per epoch; most empty chips are skipped. **`infer-gent` still walks all 133k.** Resumed from `latest.pt` so weights and optimizer continued; only the sampler changed.

**Also asked, and deferred.** An **unsupervised** encoder or classifier as extra input might help sparse geometry later (Flanders-scale pretrain). It does not fix stake labels, clusters, or diluted positives. Ranked behind this run, a height channel, and the loss change in §18.

**For write-up:** Train sampling and infer coverage are separate. Canopy v2 can still discover unlisted trees at infer time without labelling every empty cell as a negative.

---

### 17. Reading the run: curves, GIF, network size (11 Nov midday)

**Confidence GIF.** From the last full infer (`gent_city_gent_canopy_v2.json.gz`, 10,686 chips, floor 0.20): `scripts/render_confidence_map_gif.py` → `exports/inference/gent_confidence_20_40_loop.gif`. Loop **20% → 40% → 20%**. Counts: **46,760** preds at 0.20, **16,333** at 0.30, **981** at 0.40. That spread is calibration and peak shape, not proof the CNN is too small.

**`loss/pos`.** A **slight downward trend when smoothed** is normal and was what we saw (~1.04 → ~0.99 over ~40k steps). The raw curve stays jumpy. **`loss/epoch` / `loss/train_step`** (~0.00039) barely moves because the focal loss is averaged over the whole grid. Empty batches and unknown masks keep `loss/neg` near **5×10⁻⁵**.

**Bigger network?** Not the next lever. `base_channels: 32` (~475k weights) is small on purpose. Widen it only if we are still underfitting **after** the loss fix and more epochs. Map failures (silent clusters, 0.20 vs 0.40) are targets and peak rules first.

**Epoch timing.** Resume at 11:32 restarted the data loader while keeping the step counter, so this pass ended at step **42,250** (15:17), not at 33,250. **Epoch 1 loss 0.000394** became `best.pt`. A bit higher than the old ~0.00033 on 10.7k chips is expected (new σ, mixed chips, fresh AdamW). **Wait for epoch 2** (and ideally 3) before infer; a city infer would fight the GPU. Epoch 2 is another ~**3 h 40 min** (~step 75,500).

---

### 18. Validation F1 and per-tree peak loss (11 Nov afternoon) — next agenda

**Validation F1.** F1 here is the balance of precision and recall at **3 m** (strict, or **audit** which does not count distant discoveries as false positives). **Validation** means that score on a **held-out** map area, so we are not picking a checkpoint that only memorized the training streets. Training does **not** compute it. `best.pt` is the lowest **train** loss. We do **not** need F1 inside every step. We **do** need one fixed comparison before trusting this run over the archive: same small chip set or sheet, new `best.pt` vs `archive/pre_full_grid_2025-11-11/best.pt`, audit F1 plus a look at large trees.

**“Every register tree must be detected” in the loss.** Too strict. Clusters, stake-vs-crown offset, thin LiDAR, and 2 m NMS make a hard 100% recall constraint fight the data. The current loss already marks stake centers (`target ≥ 0.99`) as positives, but those pixels are diluted by ~30k cells, so a missed tree barely moves the average.

**Decision:** **per-tree peak loss** is implemented (`per_tree_peak_loss`: focal loss on the max prediction within **2 m** of each target peak, averaged over trees, added to the grid focal loss). The process that is already running does **not** use it. After **epoch 2** of the old loss, the watcher restarts with `--resume`. Config sets `per_tree_weight: 0.01` (weight **1** would be hundreds of times the grid loss and would erase the Gaussian shape, including variable σ) and `resume_weights_only: true` so Adam starts fresh. Do not compare the new epoch loss to **0.000394**.

**Watcher:** `scripts/switch_after_epoch.py` copies `latest.pt` as soon as the epoch-2 loss is finite, because the next periodic save (~500 steps later) writes epoch 2 again with an empty loss. If that window is missed, it still switches once the step counter is past **76,000**. After the kill it restores the epoch-2 snapshot so a few old-loss steps cannot replace it.

**Still later for this detector:** learning-rate drop if epoch loss flattens; the small held-out infer above; a height or CHM channel and a wider net only if the map is still weak.

**For write-up:** Sampling fixed *which chips* teach trees. Per-tree loss fixes *whether each labelled tree* can move the weights. Validation F1 is how we will know the map got better.

---

### 19. Second network on candidates (11 Nov, ~16:25) — after detection is solid

**Context.** You want inference output to be **detections**, not a second copy of the register. The register is already rich (`sortiment`, girth, planting, owner). The open question was whether a **neural net on the LiDAR of each candidate** could say what type it is, and whether it is a tree at all.

**Decision.** Yes, as a **second stage**, not a bigger heatmap model.

| Stage | Role |
|-------|------|
| **Detector** (this training) | Where a tree might be: point + score |
| **Candidate network** (later) | Small LiDAR crop around that point (or around a register stake): tree or not, and optionally a **coarse structure class** |

**Training positives are every eligible register tree.** For each eligible `gis_id`, crop the LiDAR around the stake and teach “this is a tree,” plus `sortiment` when it is a real label (not `Te Bepalen`). That includes stakes this detector never marked. A miss is still a tree, and those crowns are the ones the verifier most needs to have seen. The training set is the register, not the list of detections that happened to land on it (§36).

Add non-trees on purpose (GRB roofs and roads, detector peaks that are not canopy). Discoveries are not type labels until someone checks them.

**At inference,** the detector proposes points. The second network scores each point. If it falls within ~3 m of a register row, **attach** that row. Unmatched points stay discoveries with LiDAR scores only. The register is not re-learned as the species database.

**Limits.** DHMV II is winter, leaf-off. Full species or cultivar from this LiDAR is not realistic. Conifer versus broadleaf might be. One detector peak for a whole cluster cannot be split into five types by the crop model.

**When.** After per-tree peak loss, held-out audit F1, and a city infer you trust. Then: detections-only export, crop dataset from register trees plus hard non-trees, **verifier first**, coarse type only if the verifier helps.

**For write-up:** The map product is a list of candidates. The second model learns “tree” from every register stake, hit or missed. The register still labels the ones it already knows. LiDAR classification fills in “tree?” and a rough structure for the rest.

---

### 20. Discoveries, the 5 m line, and which F1 we use (11 Nov, ~18:45)

Epoch 2 of the old loss was still running (checkpoint step **72,000**, meta epoch **1**, finite epoch loss not written yet). The watcher was still waiting. These questions were about the scores, not a change to the cutover.

**Will the per-tree pass still allow discoveries?** Yes. The new term only asks for a high prediction within **2 m** of each register peak. The unknown mask is unchanged, so canopy far from the register is still not a negative. Nothing in the loss says “do not fire here.”

**Is 5 m a lot?** It is only the **audit** cutoff in `treelidarnet/metrics.py`, not a training radius.

| Distance to the nearest register tree | What we call it |
|----------------------------------------|-----------------|
| **≤ 3 m** | A match. That register tree counts as found. |
| **3–5 m** | A near miss. Still a false positive, even on the audit score. |
| **> 5 m** | A discovery. Audit F1 does not punish it. Strict F1 still does. |

For a street tree, 5 m is a modest gap (spacing is often **8–12 m**), so a ring 6 m from every stake is a fair separate candidate. For a wide crown it is tight: the brightest LiDAR point can sit **4–8 m** from the stake, and that same tree then looks like a near miss or a discovery next to an “unfound” stake. The **2 m** training window and **2 m** NMS do not use this 5 m rule. The map still draws every ring. We left the constant at **5.0**.

**Are audit F1 and strict F1 used?** Both are computed when inference or eval matches predictions to the register. **Neither** is computed during training, and **neither** chooses `best.pt` (that file is the lowest train loss).

- **Audit F1** is the primary comparison. `eval.py` writes it as `tree_level_audit`. City infer prints it and stores `f1_audit_at_3m`. The Ghent viewer shows it as **audit F1**. Discoveries beyond 5 m are not false positives here.
- **Strict F1** is the secondary number in the same pass (`f1_at_3m`, `tree_level_strict`, shown as **strict**). Every unmatched ring counts, including discoveries.

Those map numbers are frozen at the export floor (**0.20** on the last city infer). Moving the slider does not recompute them. The held-out check before the next full infer still looks at **audit F1**, with strict kept beside it, plus a look at large trees and clusters.

**For write-up:** Training optimizes heatmaps, not F1. Audit F1 is how we will say whether the map got better without calling every unlisted tree a mistake. Five meters is that “unlisted” line, and it is kinder to street spacing than to large crowns.

---

### 21. Epoch 3 NaN and the restart (11 Nov, ~19:58)

The watcher did its job at **19:07**. Epoch 2 closed at **0.000366** (step **75,500**) and was archived in `archive/pre_per_tree_epoch2`. The new process logged per-tree weight **0.01**, a fresh Adam, and epoch 3.

For about **1,010** steps the new term was doing what we wanted: `loss/tree` fell from about **0.47** to **0.23**, and the total loss sat near **0.004** because it now includes `0.01 × loss/tree` (do not compare that to **0.000366**). `loss/neg` rose about twenty-fold: the heatmap got hotter off the stakes as well.

At step **76,511** the loss was **NaN** and training stopped. The last logged step was still finite. Reproducing it: once the sigmoid saturates, float16 stores the score as **exactly 1** (`1 - 1e-4` is not representable in fp16). Then `-log(1 - pred)` in the negative focal term is NaN. That is the crash. A missed stake also has a large gradient (L2 around **20** versus about **0.05** on a normal batch), so a clip is there for the next spike.

**Decision.** Compute every loss in **fp32**, and clip the gradient at **5.0**. Weight stays **0.01**. Restart from the epoch-2 snapshot, not from step **76,500** (those thousand steps were the run that was about to overflow). `best.pt` was never overwritten; it is still epoch 2.

**For write-up:** The per-tree term started working immediately. The stop was float16 rounding, not the idea of the loss.

---

### 22. Unknown band 12 m → 5 m (11 Nov, ~20:20)

Front-garden trees often sit a few meters behind a street tree. Inside **12 m** the loss treats that as “not a tree.” That fights the point of this model, which is to discover unlisted trees.

**Decision.** Shrink the disk to **5 m**. The stake’s Gaussian is at most **2.5 m** wide and the per-tree term only pulls within **2 m**, so 5 m still stops one street tree from smearing into two peaks. A garden tree about **6 m** or more from the stake is now unlabeled. Training was paused at step **76,992** with the optimizer saved, **10,686** register chips were rewritten in place (empty chips were already fully unknown), and the run resumed with Adam kept. Heatmaps and LiDAR were not touched.

**For write-up:** The discovery zone now starts just outside the stake’s own crown, not a street-spacing away.

---

### 23. Reading epoch 3 after the 5 m patch (11 Nov, ~22:05)

About halfway through the pass that resumed at step **76,992** (step **~93,400** of **33,250** steps in this pass). Pace about **0.38 s/step**. This epoch should finish around **23:50**. The whole schedule through epoch 10 is still about a day after that. No NaN since the fp32 restart.

Medians across the four quarters of this pass so far:

| | First quarter | Latest quarter |
|---|---|---|
| `loss/tree` | **0.075** | **0.044** |
| Total loss | 0.0019 | 0.0015 |
| `loss/neg` | 0.00056 | 0.00045 |
| `loss/pos` | **0.69** | **0.74** (last few hundred steps ~**0.79**) |

`loss/tree` is the score that matters. A median of **0.044** is a best-in-2 m confidence of about **0.67**. The fast drop was early; it is still easing down. The total follows it, because most of that number is `0.01 × loss/tree`. Do not compare it with epoch 2’s **0.000366**.

`loss/neg` is drifting down. The heatmap is not getting hotter away from the stakes as this pass goes on.

`loss/pos` is creeping up. It only scores the exact stake cell. `loss/tree` scores the best cell within **2 m**. The peak inside the window is getting stronger, and some of that strength is sitting beside the stake. A heavy smooth on `loss/pos` hides the turn and makes it look like it is still following `loss/tree` down.

**Decision.** Keep training. Do not change the loss or the band in the middle of this pass. The map check is still a held-out infer after the epoch, not these curves.

**For write-up:** The per-tree term is doing what it was added for. The stake cell and the nearby peak are no longer the same story.

---

### 24. How far this model can go, and what it never sees (11 Nov, ~22:12)

`loss/tree` median across this pass, in blocks of 2,000 steps: **0.085 → 0.066 → 0.056 → 0.051 → 0.049 → 0.046 → 0.044**, then flat. A median of **0.044** is about **0.67** confidence within 2 m of a typical stake. The mean is stuck near **0.10**. The easy trees are fitted. The hard tail (quiet clusters, peaks far from the stake) is not catching up.

**Decision.** Finish this epoch (around **23:50**). One more epoch might shave a little. Running on to epoch 10 is the same grind, not a better model. The errors left are ones this loss does not see.

**A new loss or a new input channel does not mean starting over.** A new loss loads this checkpoint as-is; reset Adam only if the new term dwarfs the old one. A new channel means widening the first convolution, copying the current four filters, and fine-tuning. Everything after that layer loads unchanged. A wider U-Net does **not**: `base_channels` is **32**, so the stages are **32 / 64 / 128**. Doubling that is about four times the weights and a different file.

The U-Net is a shallow image model. The 200×200 grid (0.25 m) is pooled twice, to **100** then **50** cells, so the bottom cell covers **1 m** and can see a crown. Skip connections paste the fine maps back on the way up. A 1×1 head makes the score. It never receives a point.

Each chip already stores every return as local **x**, **y**, and **height above ground**, plus intensity, return number, and class. Training averages every point in a cell and drops the height. A ground hit and a branch hit in the same 0.25 m become one number.

**Decision.** Do not rebuild the detector as a point network. Leaf-off aerial LiDAR will not show a botanical crown. It will show a column, a spread of high points, and a gap above the ground, which is enough to separate a shrub, a pole, and two crowns, and which the average erases. Next representation change, after this epoch’s map check: a few height summaries per cell (highest point, spread, share of points above a few meters), fine-tuned from these weights. The crop model (§19) is where the points are compared with each other.

**For write-up:** The map model sees a flattened photograph of the points. The shape of the cloud is saved in the chip and deliberately not used yet.

---

### 25. A 3D picture of the LiDAR crop for Ghentree (11 Nov, ~22:21)

The crop of points around each detection is what the second model will see (§19). You want that same object kept so it can be shown later on **Ghentree** (`~/Dev/HugATree`) as the points themselves: a **static isometric image**, or an **interactive 3D** view of the cloud.

The city map stays the Deck.GL circles it already has (`ScatterplotLayer` in `static/js/map.js`). The point picture is the detail of one detection, opened from a ring. It is the winter scan of that object: trunk, branches, and whatever crown the leaf-off flight actually hit. Not a smoothed tree icon, and not a column built from a few radii.

**Decision.** Save the crop with the detection, because that is the input of the second model and the source of the picture. For the website, two weights of the same crop:

| Form | What the visitor gets |
|------|------------------------|
| **Static isometric image** | One small picture per detection, rendered ahead of time from the points. Lightest. A thumbnail on the tree panel. |
| **Interactive 3D** | The same points, thinned, in a viewer that opens for the tree you clicked. Not drawn for every ring on the city at once. |

The full-density crop stays on disk for training. The page loads either the image or a thinned cloud for one tree. Discoveries and register trees use the same picture, so an unlisted tree can be shown even when it has no `sortiment`.

**For write-up:** The second model and the website share one crop. The model scores the points. The site shows them, as a still isometric frame or as a small 3D view.

---

### 26. How long the next infer takes (11 Nov, ~22:24)

The slow part is turning each chip’s points into the 200×200 grid. That scatter is a Python loop over the points, and `infer-gent` loads chips with no worker processes. A 24-chip sample took **8.76 s**, about **0.37 s per chip**. The U-Net forward pass is a small fraction of that.

| Run | Chips | Time |
|-----|-------|------|
| Smoke | a few hundred | a few minutes |
| Register chips only | **10,686** | about **1 hour** |
| Full index (the default) | **133,000** | about **13–14 hours** |

**Decision.** Leave infer until this training process is off the GPU. A held-out sample is minutes. The full city map is an overnight run, and only after this checkpoint is the one we want to publish.

**For write-up:** Inference time is the point-to-grid step, one chip at a time.

---

### 27. Infer from the center of Ghent outward (11 Nov, ~22:27)

`infer-gent` walked chips in filename order. That is not west-to-east, and it is not the city. The live map writes partial rings as chips finish, so the order is what you see appear.

**Decision.** Sort by distance from the middle of the chip grid (about **103.8 km E, 196.0 km N** in Lambert). Then split that order: chips with a register tree first, chips with none after. Each group still grows outward. `--limit` keeps the central register chips. The finished detection list does not change. Training still uses its own sample order.

**For write-up:** A live infer fills the middle of Ghent with register chips first, then the empty tiles, still from the center out.

---

### 28. What `loss/train_step` is (11 Nov, ~22:34)

`loss/train_step` is the total loss of one batch of 4 chips, written after every optimizer step. The printed epoch loss is the average of these step values.

On this pass that number is the grid focal loss plus **0.01 × `loss/tree`**.

The other curves are pieces of the same batch. `loss/pos` is only the stake cells. `loss/neg` is only the negative cells. `loss/tree` is the per-tree term before the 0.01 weight. A smooth of `loss/train_step` mostly follows `loss/tree`, because `0.01 × 0.044 ≈ 0.00044` is a large share of a total that sits near **0.0015**.

The same scalar near **0.00039** in §17 was the grid term alone, on the epoch-1 scale. A value near **0.001–0.002** now is the new total. That is the same reading as “How to read the loss” below.

**For write-up:** `loss/train_step` is the epoch loss before it is averaged. It is the quantity that changed scale when the per-tree term was added.

---

### 29. Heatmap peaks, and whether 0.25 m is enough (11 Nov, ~22:37)

The network writes one score, from 0 to 1, on every cell of the 200×200 map. A tree is a local high spot on that map.

A cell is kept when its score is at least **0.20** and it is the highest score inside a **2 m** neighborhood. Those local peaks are sorted, at most **64** are kept, and a second **2 m** pass drops any peak that sits next to a higher one. The cell center is the map position. Overlapping chips are merged at **2.5 m**, and the higher score wins.

Training asked for a Gaussian mound around the stake. Inference reads the top of each mound.

**Is 0.25 m a fair resolution for individual trees?** Yes, for placing a tree. The winning cell’s center can be off by at most about **12 cm**. Stakes, crown peaks, and the **3 m** match are measured in meters. Street trees are often **8–12 m** apart, which is many cells.

What decides whether two trunks stay separate is coarser than the grid. The training mound is **1.0–2.5 m** wide, the deepest U-Net layer sees **1 m** cells, and any weaker peak within **2 m** of a stronger one is dropped. Two trunks inside that 2 m come out as one detection. The LiDAR is about **16 points per square meter**, about one return per 0.25 m cell, so a finer grid would be mostly empty and would still be scored against the same mound and the same 2 m rule.

**Decision.** Leave the grid at **0.25 m**. Separating closer trees means a change to the mound, the suppression radius, or a later instance model.

**For write-up:** The heatmap is fine enough to point at a tree. Splitting a cluster is a different lever.

---

### 30. Clear the ignore ring (11 Nov, ~22:55)

A missed register stake is punished by the per-tree term, one vote per stake, best score within **2 m**. The grid average barely notices that one cell. Unlisted trees stay unpunished on purpose.

The ignore mask does not. It zeros the loss on a ring **8–15 m** in from the tile edge, about **30%** of each chip. Tiles meet edge to edge, so a stake in that ring is never a positive anywhere. The spec called that incomplete context. The cut-off part is the outer **8 m**, and that part was already trained.

**Decision.** `ignore` is all zeros from here on. `data-patch-chips-ignore` rewrites the stored chips. Do not run it while this epoch is reading them. When the epoch exits, patch, then train **one more epoch** so those stakes get a loss. This epoch’s infer still reflects the hole.

**For write-up:** About a third of register stakes were invisible to the loss. The next epoch is the one that can see them.

---

### 31. Faster infer, same grids (11 Nov, ~23:15)

The **13–14 h** in §26 was a Python loop that added one LiDAR point at a time, on the main process. Replacing that loop with one sum in the same point order matches the old cell values exactly. On a chip with 28,000 points the loop took **0.55 s** and the new sum **0.001 s**.

`infer-gent` now builds the next chips on **4** workers while the GPU runs the current batch (`--num-workers`). It reads points, features, and origin only. The heatmap and the masks stay in the file. Chip order, scores, and peak positions are unchanged. Half precision and skipping empty chips were left out, because those would move scores.

A fresh 24-chip sample, grids only, took **0.11 s** (**0.005 s/chip**). The old overnight budget does not apply. The forward pass is what is left, and it was the small slice of the old 0.37 s. The first progress lines of the real infer are the timing to trust.

**For write-up:** The map is the same function of the same grids. The hours were the loop.

---

### 32. The center is full of rings (12 Nov, ~00:15)

Epoch 3 finished. The printed loss is **0.0021**. That is the new total (grid focal loss plus `0.01 × loss/tree`), at step **110,242**. Training was stopped there. `best.pt` is that file. The epoch-2 archive is still the sparser map.

A live look at **400** central chips wrote **20,398** rings, about **50 per chip**. The median score is **0.72**. Nothing sits closer than **2 m**, so NMS is doing its job. The volume is real peaks.

In that same **1.7 km** box the register has **3,981** trees. At the **0.20** floor, **2,548** rings are within **3 m** of one of them and **16,080** are farther than **5 m**. Sliding the floor up:

| Floor | Rings | Per chip | Share more than 5 m from a local register tree |
|------:|------:|---------:|-----------------------------------------------:|
| 0.50 | 20,325 | 51 | 79% |
| 0.70 | 11,822 | 30 | 73% |
| 0.80 | 4,125 | 10 | 57% |
| 0.85 | 2,138 | 5 | 43% |
| 0.90 | 1,042 | 3 | 29% |
| 0.95 | 349 | 0.9 | 18% |

The map looks reasonable around **0.85–0.90**. Real stakes score higher than the extra bumps. The extras are still confident, because cells more than **5 m** from a stake have no negative loss except on GRB. The unknown mask did what it was built to do: it allowed discoveries, and this checkpoint used that freedom everywhere.

**0.85 is a way to look.** It is not a new training target. Another epoch of this same loss will stay hot. The next training change is a light penalty on peaks in the unknown area, strong enough that a field of **0.7** bumps loses, weak enough that a real unlisted crown can win. Before that epoch, patch the stored ignore ring so every stake in the tile is a positive (§30). Then look at the center again at **0.20** and at **0.85**.

Tonight’s full infer is live, all **133,000** chips, export floor **0.20**, slider opening at **0.85**. Amber rings draw on top of the green discs. The low audit F1 on the 400-chip file used all **59,619** register trees as the denominator, so it is not a city score.

**For write-up:** The ranking works. The published floor of 0.20 does not, on this checkpoint. The calm comes from a penalty on unknown peaks, not from hiding the file.

---

### 33. Stop the hallucinations (12 Nov, ~00:17)

The volume in §32 is the problem to fix. This model has to stop inventing trees.

A ring on a real crown that the register never listed is still a discovery, and those stay allowed. A ring on a bump with no crown is a hallucination. Epoch 3 does the second thing across the center: tens of thousands of confident peaks, most of them more than **5 m** from any stake. The slider at **0.85** only hides them. It does not teach the network.

The next epoch is for that. Patch the ignore ring, then train one epoch from these weights with a penalty on peaks in the unknown area. Empty ground should lose. A real unlisted crown should still be able to win. Success is the center at the **0.20** floor looking like a tree map, not a carpet that only looks fine once the slider is pushed up.

**For write-up:** Discoveries stay. Hallucinations have to go. That is the point of the next epoch.

---

### 34. Keep the trees the high slider already found (12 Nov, ~00:21)

The carpet and the promising map are different rings. On the live file, peaks from **0.50 to 0.85** pile up on the chip frame: about **40%** sit within **4 m** of a 50 m edge, against **29%** of the area. The hottest cells are **2–5 m** inside the corners. Four chips meet at each corner and the merge radius is **2.5 m**, so those peaks stay as a little cluster. That is the grid.

From **0.90** up the frame is gone. Edge share **29.6%**, corner share **4.8%**, inner share **15.7%**, each in line with the area. Those rings are spread like trees. On the 400-chip center, the same high floors were the ones that sat on the register (§32).

The next penalty has to take down the **0.5–0.85** field and that corner lattice. It has to leave the high peaks standing, including a high peak on a crown the register does not list. Distance to a stake is the wrong knife. A real unlisted tree is far from the register too.

The check after that epoch: rings per chip at **0.20** fall, and the register trees this checkpoint already marks at **≥ 0.90** are still detected.

**For write-up:** The high slider is the set to protect. The penalty is for the repeated mid-score frame, not for every ring the register missed.

---

### 35. The east–west streaks are a flight line we never downloaded (12 Nov, ~00:36)

Some rows of chips have no amber rings even though the green register trees are there. Those chips were built with no points. **25,979** chips have `point_count == 0`, and **1,841** of those still contain register trees. The model is scoring an empty grid.

The tiles are named on a **500 m** grid. About one in five files stops **40–200 m** short of its northern edge, and the next file starts on the **500 m** line, so the strip between them is empty. Just south of the center, flight `22011_2` at corner `103500_195500` ends at y = **195,962**. The same corner has another file, flight `22010_2`, and that is the one that covers the rest of the square. About **400 m** north, flight `22009_6` stops near **196,382** and flight `22006_2` is the file for the rest.

The downloader asked OpenLiDAR for one point in the middle of each square and saved that file. The flight line that only covers the edge was never requested. `index_laz_files` also keeps a single path per corner, so tomorrow both files have to be read. Different flight ids already mean different filenames, so a second download will not overwrite the first.

**Tomorrow**, not during this infer. Download the other strip for each square, then rebuild the chips that have no points. Do that before the next epoch, so those register trees are in the LiDAR the model sees. Leave the chips alone while this infer is reading them.

**For write-up:** The streaks are missing flight lines, not a silent model. The register was never the source of the points.

---

### 36. The second model learns from every register tree (12 Nov, ~01:06)

The crop dataset is built from the register, not from the detections.

Every eligible stake gets a LiDAR crop labeled “this is a tree,” whether or not epoch 3 put a ring on it. The trees the detector misses — large crowns, clusters, a peak shifted off the stake — are part of that set. Training only on registry hits would teach the easy street trees and skip the crowns the first model fails. At inference the second model still runs only where the detector proposed a point. A miss can be in the training set and absent from the map until a later detector finds it.

Non-trees are added on purpose: GRB roofs and roads, and detector peaks that are not canopy. A discovery stays unlabeled for type until someone checks it. Coarse structure only, after the tree-or-not call works (§19).

**For write-up:** Detections choose where the second model is asked. The register, all of it, chooses what it studies.

---

### 37. PointNet++ crop verifier (12 Nov plan; 16 Nov — skip plain PointNet)

The city detector stays the BEV U-Net (§24). The second model is the one that sees the points.

A crop is a vertical stack. Near the ground the returns are a narrow column or a gap. Higher up they spread into a crown. **PointNet++** with two set-abstraction levels—a tight neighborhood for the column and a wider one for the crown—then one **tree-or-not** score. Coarse type, later, is the same backbone. Conifer versus broadleaf in this leaf-off scan is mostly that profile.

**Order (16 Nov).** Implement **PointNet++ first**, not a staged plain PointNet. The column/crown signal is the whole point of the verifier; anti-carpet on the BEV already showed that score-only fixes do not separate lattice junk from real canopy. Optional: a height-stats MLP on the same crop index as a cheap ablation baseline—not the shipped verifier.

**Training data (§36).** Every eligible register crop plus GRB roofs/roads and detector peaks where **`hard_negative`** is hot at the peak—not carpet or off-register mid scores.

**Crop.** Center x and y on the stake or the detection. Use height above ground, plus intensity, return number, and class. Subsample to a fixed count, on the order of **512** points. Ball radii are in meters. A Point Transformer is the wrong size for about **59,000** sparse crowns and a yes-or-no label. Runs only on proposed points at infer. Full spec: [docs/SPEC.md](docs/SPEC.md) §6.5.

**For write-up:** The map model is a photograph of the points. The classifier is a small **two-scale PointNet++** on one tree’s cloud.

---

### 38. The other flight line is on disk, and mid-score peaks get pushed down (12 Nov, morning)

OpenLiDAR returned the second strip for each **500 m** square. **1,864** new files, **1,330** squares. The index now keeps every file for a corner. The center gap chip `gent_50m_103700_195950` went from **4** points on flight `22011_2` to **28,626** on `22010_2` plus those 4. The **25,979** chips that were stored empty are being rebuilt from those files before the next epoch.

The penalty is the mean score of unlabeled local maxima in **[0.20, 0.90)**, inside the same **2 m** window inference uses, weight **0.1**. A local maximum at **≥ 0.90** is not in that mean, so a sharp high peak can stay with no stake nearby. The mask is detached: a **0.7** bump is only pushed down, not rewarded for climbing to **0.90**. Distance to a stake is not in the term. The printed epoch loss will include `0.1 ×` that mean. Do not compare it with **0.0021**.

One epoch, from `best.pt` (step **110,242**), after `data-patch-chips-ignore`. Not from `latest.pt`. Export floor stays **0.20**.

**For write-up:** The streaks were a missing flight line. The carpet term lowers mid-score unlabeled peaks and leaves the high ones alone.

---

### 39. Anti-carpet BEV failed on the map; freeze epoch3_hot; PointNet++ verifier next (12 Nov, late morning)

**What we tried.** After the second strip rebuild and `clear_stored_ignore` in training, we added `unknown_peak_loss` on unlabeled local maxima and ran resume epochs from epoch-3 weights:

| Run | Config | Outcome |
|-----|--------|---------|
| **epoch4_flat** | weight **0.1**, band **[0.20, 0.90)**, mean of mid peaks | One full pass (~110k→143k steps). Printed loss **~0.021** (new scale). Live infer: **every score ~0.20**, **zero ≥ 0.90**, ~**700** preds vs **~1.6M** on epoch3 — **heatmap collapsed** to the export floor, not an infer bug. Archived: `archive/epoch4_flat/`. |
| **epoch4b** (anticarpet) | weight **0.01**, band **[0.45, 0.88)**, focal on punished peaks; focal in [`treelidarnet/models/losses.py`](treelidarnet/models/losses.py) | Interrupted then resumed from `latest` @ **122k** → **155250**. `best.pt` promoted at printed **0.0039**. |
| **Center QA** | `qa-checkpoint-center` on 400 central chips | Rings/chip @0.20: **59.2 → 58.0** (slight win). High-conf recall @0.90 (stakes that were ≥0.90 on baseline): **47/932 (~5%)** — **FAIL**. Reverted `best.pt` to **`epoch3_hot`**; candidate in `archive/epoch4b_qa_fail/`. |

**Decision.** **Pause further anti-carpet BEV fine-tunes** until a recipe passes center QA (rings down **and** high-conf recall ≥ ~98%). **Ship the map on `epoch3_hot` only.** Export floor stays **0.20**; live slider **~0.85**. Mid-score carpet mostly sits **below 0.90**, so the high view is usable while we add a second stage.

**Training wall clock vs steps.** A full index epoch is still **~33k** optimizer steps. Late epoch-3 **hours** were mostly **DataLoader / disk stalls and pauses** (patch, NaN restart), not slower GPU steps (~**17–21 ms** when the loop moves). Fast passes today are **uninterrupted stepping** with warm page cache — not shortened epochs.

**Pivot: PointNet++ crop verifier (§36–37).** Train a small **tree / not-tree** classifier on **LiDAR crops** (~**512** points). **Positives:** every **eligible** register stake (including detector misses). **Negatives (v1):** GRB **building/road** crops and detector peaks only where the chip **`hard_negative`** mask is hot at the peak — **not** generic carpet or off-register mid scores labeled “not tree,” because many of those are **real unlisted canopy** (discoveries). At infer, run the verifier **only on detector proposals**; track **discovery retention**, not just FP drop on roads.

**Ops added:** [`treelidarnet/qa_center.py`](treelidarnet/qa_center.py) + CLI `qa-checkpoint-center`; [`configs/gent_canopy_v2_anticarpet.yaml`](configs/gent_canopy_v2_anticarpet.yaml) for experiments; resume/archive helpers in [`treelidarnet/train.py`](treelidarnet/train.py).

**For write-up:** The BEV detector is **good enough to freeze** at epoch3 for product QA; **carpet is a presentation + verifier problem**, not another blind BEV epoch. Labeling carpet peaks as negatives would teach the verifier to **kill discoveries**.

---

### 40. Crop QA viewer, training hygiene, and verifier v2 (16 Nov, midday)

**Your questions.** Register trees **pop in and out** when zooming. The **3D crop panel** failed to load. After starting train: **is the loss function correct for our use case?** **TensorBoard** did not show anything.

**Crop map (HugATree pattern).** The bug was refetching the register on every `moveend` with a bbox and **`limit=8000`**, so the visible set changed with zoom. Fix: one **`GET /api/register/all`** (~59k eligible stakes), cached on the server — same idea as HugATree’s single `data.json` load. Predictions still come from one gzipped export; clicks fetch `/api/crop` for the 512-point tensor.

**3D panel.** OrbitControls import bare `three`; the browser needs an **import map** in `crops.html`. WebGPU needs **`renderAsync`**, not `render()`, with WebGL2 fallback. TensorBoard is **not** started by `train-crop`; run **`make tensorboard`** (or port **6006**) separately.

**Loss review — what was wrong vs what was fine.**

| Piece | Verdict |
|-------|---------|
| **BCE on logits** + sigmoid at infer | Correct for binary tree / not-tree |
| **Validation** | **Broken proxy:** all GRB negatives were `split=train` → val was almost only positives; low val loss could mean “always say tree” |
| **Class balance** | ~**4:1** pos:neg with unweighted BCE → gradients favor positives (bad for an FP filter) |
| **Failed crop extract** | Dataset filled **zeros** but kept the label; infer returns score **0** — inconsistent |
| **`best.pt` rule** | Lowest val BCE tracked the wrong goal |

**Decisions (implemented).**

1. **Manifest:** GRB negatives use the same **85/10/5** hash split on `crop_id` as register trees (`_split_bucket`). Rebuilt index: **73,697** rows — val **7,498** (**1,522** neg), test **3,686** (**708** neg).
2. **Loader:** `None` on failed extract; collate drops bad rows (no phantom trees).
3. **Train:** `balance_class_loss: true` → BCE `pos_weight = n_neg / n_pos` (~**0.25**). Log **precision / recall / F1 / AUROC** and mean prob on pos vs neg; promote **`best.pt` by val F1**.
4. **Run name:** **`gent_crop_verifier_v2`** so logs do not overwrite the naive v1 attempt; config still `configs/gent_crop_verifier_v1.yaml`.

**Project outline (where this sits).**

```
LAZ → chips (BEV) → U-Net proposals (frozen epoch3_hot)
                         ↓
              detections @ score ≥ 0.20 (export); map slider ~0.85 for QA
                         ↓
        PointNet++ on 8 m / 512-pt crop → verifier_score → filter FPs
                         ↑
   trains on ALL eligible register crops + GRB hard_negative crops (NOT carpet/discovery as label-0)
```

**Still open after v2 trains.**

- Offline **`filter-export`** on a center city bundle; measure **discovery retention** (SPEC §6.5).
- Threshold sweep on val, not fixed **0.5** at infer.
- Detector-peak negatives where **`hard_negative` is hot** (v1 manifest is GRB cells only).

**For write-up:** We caught a classic trap — **imbalanced val and checkpoint metric** — before treating TensorBoard loss as “the model works.” The crop QA viewer is the human loop for “does this cloud look like a tree?” beside the amber ring.

---

### 41. Verifier-only RGB from DHMV II (16 Nov, afternoon)

**Your question.** DHMV II LAZ already has **red/green/blue** (leaf-off or not, crowns look “real” in CloudCompare). Must we rebuild every BEV chip and retrain the U-Net?

**Decision.** **No.** Keep the BEV at **`in_channels: 4`** and frozen **`epoch3_hot`**. Put RGB on the **crop verifier** only:

| Layer | Channels | RGB source |
|-------|----------|------------|
| BEV detector | 4 (agg feat + density) | unchanged — chips still store 3 LiDAR attrs in `feat` |
| Crop verifier | **9** = centered xy + h_agl + intensity/return/class + **RGB** | `rgb` array patched into npz from LAZ |

**Why patch, not rebuild.** Training touches **~16.8k** unique chips in `crops_index.parquet` (73.7k crop rows), not all **133k** grid tiles. `data-patch-chips-rgb` re-reads LAZ per chip bbox, matches points to existing `xyz` (same h_agl keys as build), writes `rgb` (N,3) beside `feat`. No heatmap/GRB rewrite. Full-grid infer later needs **`--all-chips`** if we score detections outside the manifest chip set.

**Probe.** OpenLiDAR DHMV II format **3** includes `red/green/blue` (uint16, typically 10k–60k on vegetation).

**Code.** `treelidarnet/data/laz_io.py` (optional RGB in bbox read), `patch_chip_rgb.py`, `crops.py` `IN_CHANNELS_RGB=9`, `configs/gent_crop_verifier_v3_rgb.yaml`, 3D viewer uses API `red/green/blue` when present.

**Train path.** After the manifest patch **walk** (not necessarily 100% on-disk success): `train-crop --config configs/gent_crop_verifier_v3_rgb.yaml` → `runs/gent_crop_verifier_v3_rgb/`. Crops on chips without `rgb` are skipped at load time. v2 **6-channel** checkpoint does not load into 9-channel weights without a fresh run.

**For write-up:** Color is a **second-stage** cue (crown vs roof tint, column vs spread) that the BEV deliberately throws away when it rasterizes. We did not reopen the anti-carpet epoch or multimodal BEV scope — only the PointNet++ crop that already sees points.

---

### 42. Full-city BEV @ 0.80, infer resume, COPC patch bug, verifier priority (16 Nov, late afternoon)

**Full infer @ 0.20 + `--live`.** First **133k** pass **OOM’d ~49.2k** chips (~2.8M preds). Restarting without resume **wiped** live `predictions.json`. Code fix: **`--resume-live`**, incremental live merge (flush **new** chips into saved preds), no status reset on resume.

**Full infer @ 0.80, no live.** Completed **133k** chips → **`gent_city_gent_canopy_v2_full_080.json.gz`**: **~1.17M** preds (extract floor = threshold). Audit F1 **~0.67** on bundle (strict F1 low — many discoveries). **0.20 city live infer abandoned** for production: too many peaks for RAM and the map.

**Default export.** Center **~10.7k** file archived → `exports/inference/archive/gent_city_gent_canopy_v2_center_10kchips_20251112.json.gz`. **`gent_city_gent_canopy_v2.json.gz`** is now the **full 0.80** city (copy of `_full_080`); `ghent_map.js` / crop QA default path unchanged.

**Stage-2 live on full city.** Started **`filter-export --live`** on **1.17M** detections (v2 verifier). **Paused** after ~1k scores — competes with chip I/O and was queued behind RGB + v3 train. Scoring **1.17M** crops is **GPU-hour scale** (batch it; not the same job as LAZ RGB patch).

**RGB patch ops.** Crashed at chip **~3500**: `*.copc.laz` sidecars indexed as `*.laz` — invalid LAS signature. Fix: **exclude `.copc.laz`** in `index_laz_files`; skip bad paths in `read_laz_bbox`. Resume with `skip_existing`; chunked LAZ read + **`--jobs 1`** on laptop RAM.

**RGB patch — two scopes (do not conflate).**

| Scope | Command / list | Status (16 Nov evening) |
|-------|----------------|-------------------------|
| **Crop manifest** (verifier **train** + crop QA) | Default `data-patch-chips-rgb` → unique `chip_id` in `crops_index.parquet` (**16,843**) | **Walk finished** (`reel_then_rgb.log` **15:25**): last run reported **3,041** new writes; **earlier run + skips** mean on-disk is higher |
| **Full city grid** (verifier on **arbitrary** detection chips) | `--all-chips` + `--only-missing-rgb` → `chips_missing_rgb.parquet` (**~92.8k** missing of **133k**) | **Paused ~19:34** at **~6.3k / 92.8k** (`patch_chip_rgb_resume.log`) — deprioritized for register train gate |
| **Register-hosting tiles** (v3 **train** gate) | `--register-chips --only-missing-rgb --refresh-missing-manifest` → **`chips_register_missing_rgb.parquet`** (**2,166**) | **In progress** (`patch_chip_rgb_register.log`) |

**On-disk check (mmap `data/processed/chips/*.npz`, 16 Nov ~19:10):**

| Chip set | Unique chips | Valid `rgb` (len = `xyz`) | Still missing |
|----------|--------------|---------------------------|---------------|
| All manifest chips (`crops_index`) | 16,843 | **13,522** (~80%) | **3,321** |
| **Registry-containing** chips (tiles used by `source=register` crop rows) | 10,552 | **8,386** (~79%) | **2,166** |

**Yes — some registry-containing chips are still unpatched.** A “registry chip” here is a **50 m tile that hosts at least one eligible register tree crop**, not “every chip that ever appeared near the register.” **~2.2k** of those tiles still had no usable `rgb` before the register pass (LAZ has no color, bbox read fail, **&lt;90%** xyz match, empty tile, etc.). For **v3_rgb**, crops without `rgb` are **skipped** (not trained with zeros) — that drops **~11.9k / 58.9k** register **rows** (~20%), including **~10.1k** train positives, until those tiles patch.

**Train gate (your bar).** Start **`train-crop` v3_rgb** after register tiles are as complete as patch allows, then recount manifest loadable rows. Full **92.8k** city RGB is **not** required for that gate.

**Full-city RGB duration (corrected).** With **`--only-missing-rgb`** and **`--jobs 1`**, the resume log shows **~80%** of queued chips getting writes and roughly **~100+ chips/min** scan throughput when the job is healthy — **~12–20 h** for **~92.8k**, not multi-day wall time (unless the machine sleeps or the job restarts). Earlier “days” wording mixed up **(a)** million-detection **filter-export** and **(b)** a pessimistic guess before **`--only-missing-rgb`** and chunked LAZ.

**Map QA signal (v2).** High-BEV rings **still** pass on **building walls** in the inner city — columnar returns + GRB negatives don’t catch every façade. **Hypothesis:** RGB on the **crop verifier** (v3) helps separate **vegetation tint / crown spread** from **flat painted walls** better than intensity + class alone.

**For write-up:** Ops lesson — **long jobs need resume and backups** before restart; **extract threshold** is a first-class knob (0.80 for ship, not 0.20 + slider at city scale). **Verifier before RGB** was the wrong queue order.

---

### 43. Register-first RGB, full-city pause, v3 train gate (16 Nov, ~19:30)

**Your bar.** You want **register training crops** covered before **`gent_crop_verifier_v3_rgb`** — not “start v3 on ~80% of chips and silently drop **~20%** of register positives.”

**Mechanism recap.** `in_channels: 9` → no `rgb` ⇒ `extract_crop_from_chip_path` returns **`None`** ⇒ row omitted from batch. No poisoned gradients; **coverage** loss on register trees.

**Ops pivot.** Full-city **`--all-chips --only-missing-rgb`** was **stopped** (~**6.3k / 92.8k** scanned in `patch_chip_rgb_resume.log`) so LAZ I/O targets the **2,166** registry-hosting tiles still missing color, not arbitrary grid order.

**CLI.** **`--register-chips`**: chip list = unique `chip_id` where `crops_index.source == "register"` (**10,552** tiles). With **`--only-missing-rgb`**, manifest file = **`data/manifests/chips_register_missing_rgb.parquet`** (separate from **`chips_missing_rgb.parquet`** for the full grid).

**Command (running):**

```bash
uv run treelidarnet data-patch-chips-rgb \
  --register-chips --only-missing-rgb --refresh-missing-manifest --jobs 1
```

Log: **`data/logs/patch_chip_rgb_register.log`**. Expect **~30–90 min** for **2,166** chips (not the **92.8k** queue).

**After register pass:** mmap-count register rows loadable for v3; optional default manifest re-patch for remaining **~1.1k** non-register manifest gaps; then **`train-crop`**. Resume full-city RGB only when you need verifier color on **off-manifest** detection chips.

**Register pass 1 → 0 writes (bug).** All **2,166** attempts returned false: **3D** rounded-key match used **recomputed h_agl** vs chip `xyz` z; **~88%** matched, below the **90%** gate — so LAZ had RGB but nothing was saved. **Fix:** `_match_rgb_to_chip_xyz` falls back to **(x, y)** keys only (last-wins), then **re-run** `--register-chips` patch.

---

### 44. Dual verifier QA, full RGB, v2/v3 wall findings, discoveries filter (16–17 Nov)

**Map UX — compare v2 and v3 on the same ring.**

| Surface | Behaviour |
|---------|-----------|
| **`serve-crop-qa`** | Loads **`gent_crop_verifier_v2/best.pt`** and **`gent_crop_verifier_v3_rgb/best.pt`**; query param **`verifier_mode=compare\|v2\|v3`** |
| **Click panel** | Side-by-side **Ver v2** / **Ver v3** bars; 9-channel RGB crop when patched |
| **Halos** | Need **dual live** `filter-export`: each row gets **`verifier_scores: {v2, v3}`**; **outer blue** = v2, **inner green/red** = v3 (`map_layers.js`) |
| **Makefile** | `viewer-crop-qa`, `filter-export-live-compare` (25k cap), **`filter-export-verifier-full`** (no limit) |

**Live compare (25k pilot, BEV ≥ 0.85).** Completed dual score on **25,000** detections. On **RGB-eligible** subset (**12,226** of 25k):

| Metric | v2 | v3 |
|--------|----|----|
| Kept @ 0.5 | ~87.8% | ~87.7% |
| Register ≤3 m mean score | 0.67 | **0.74**; retention **89% → 96%** |
| Discovery >5 m retention @ 0.5 | **87%** | **81%** |
| AUROC (register vs not) | ~**0.50** | **~0.66** |

**Qualitative QA (your eyes).** LiDAR crop **looks like a tree** but **v3 rejects** vs v2 — common pattern: **~983** cases v2≥0.5, v3&lt;0.5 on RGB tiles; **~86%** are **discoveries** (&gt;5 m from register), very high BEV. **~671** have v3 in **0.35–0.5** (threshold tuning). v3 uses **RGB**; BEV does not — v3 often **downscores** LiDAR-plausible peaks when ortho does not match register-style “tree class.” **Not assumed bug** without checking RGB in the 3D panel.

**Prod direction (open).** Combined rule under discussion (e.g. v2 for discovery, v3 for register-near; OR at infer). **Do not** relabel BEV from **v3-low alone** on discoveries — same mistake as using carpet as verifier negatives (§39).

**Crop map — discoveries-only filter.** Toolbar toggle hides predictions within **min distance** of any register stake (haversine + spatial index). **Slider** when enabled; default **2.0 m** (user-tunable **1–15 m**). Independent of audit **5 m** discovery bucket but same spirit for QA.

**Full-grid RGB (17 Nov).** Resumed **`make patch-chips-rgb-full`** (`--all-chips --only-missing-rgb --refresh-missing-manifest --jobs 1`). Queue **83,290** chips; **`Done: rgb on 79,460 chips (83,290 requested)`** — remainder = no LAZ color, match fail, or empty. Register + manifest passes had already completed (**2,165** + **1,155**). v3 scoring off-manifest detections is now **much** better covered on disk.

**Full verifier “infer” (17 Nov).** User meant **`filter-export`**, not BEV **`infer-gent`**. Job: **`filter-export-verifier-full`** → **530,258** inputs @ **BEV ≥ 0.85** (not full **~1.17M** @ 0.20), dual v2+v3, streams to **`exports/inference/live/verified_predictions.json`**, output gz **`gent_city_gent_canopy_v2_verified_v2_v3_full.json.gz`**. Log: **`data/logs/filter_export_verifier_full.log`**. Wall time: **multi-day** (2× forward per detection).

**Near miss — BEV re-infer.** Assistant briefly started **`make infer-gent-full`** (same path as shipped export). Stopped **~16k/133k** chips. **`gent_city_gent_canopy_v2.json.gz`** still **Nov 16 13:39**, **~29M**, **~1.17M** preds — **`infer-gent` only writes gz after full 133k loop** (non-live).

**Weights and git.** **`runs/`**, **`*.pt`** are **gitignored** — checkpoints stay local; share via releases / copy, not repo commits.

**Asked later: train BEV on high-verifier predictions?** Possible as **sparse pseudo-labels** (consensus high on discoveries; hard neg only with GRB + dual reject). **Not** silver bullet: circular with verifier, v3 RGB cue invisible to 4-ch BEV, risks killing discoveries. Prefer **frozen BEV + dual verifier** until carpet/discovery tradeoff is nailed on **post–full-RGB** scores.

**For write-up:** Second-stage story is **two specialists** (LiDAR geometry vs LiDAR+RGB), not one score. Full-city RGB was the gate for fair v3 map QA, not a BEV retrain.

---

## Where we left off (19 Nov 2026)

**Detector (unchanged):** `best.pt` === **`epoch3_hot`**. **Shipped city export:** **`gent_city_gent_canopy_v2.json.gz`** @ **0.80**, **~1.17M** preds.

**Verifier + product QA (active):**

| Item | State |
|------|--------|
| Full dual verifier export | **`gent_city_gent_canopy_v2_verified_v2_v3_full.json.gz`** — **530,258** scored @ **BEV ≥ 0.85** (inputs to high-conf subset) |
| Explorer bundle | **`exports/inference/explorer/`** — **402,749** detections + **59,619** register rows in columnar `.bin` |
| End-product UI | **`make explorer-serve`** → **http://127.0.0.1:8765/explorer** (Makefile **`viewer-stop`** frees port from crop QA) |
| Map scores | **Precomputed** BEV + v2 + v3 from bundle — **no GPU** on pan/zoom; clicks fetch LiDAR crop only |
| FP curation | **`data/manifests/explorer_qa_flags.parquet`** — manual **`explorer_fp`** → `data-build-crop-index` → retrain v2/v3 → **re-export** |
| v2 / v3 checkpoints | Local **`runs/gent_crop_verifier_v2/best.pt`**, **`runs/gent_crop_verifier_v3_rgb/best.pt`** (not in git) |

**Next steps (ordered):**

1. **Full verifier re-export on frozen BEV** — **`make filter-export-verifier-full`** (weighted **@ 0.45**, current or `_fp` ckpts) → **`make explorer-bundle`** (§47).
2. **QA on that bundle** — flag **diverse** building FPs; use **Hide flagged** + combine sliders for end-product preview (§45).
3. **Later FP batches (§48):** `data-build-crop-index` → incremental **`train-crop`** from `*_fp/best.pt` (**5–8 epochs**) or full **15** from baseline if large refresh → **`filter-export-verifier-full`** again (same BEV gz) → bundle → **`eval-explorer-flags`**.
4. Optional: promote `_fp` paths in Makefile / **`serve-explorer --rescore-on-click`**; tighten explorer combine before changing export combo-t.
5. Optional: GRB **point-in-building** veto at export (not v1); deep-scan ROI grid (separate idea).

**BEV rule unchanged:** frozen **`epoch3_hot`** unless **`qa-checkpoint-center`** passes on a replacement.

---

### 45. Flask end-product QA explorer (17 Nov plan → 19 Nov shipped)

**Goal.** After full dual verifier scoring, a **local Flask** app replaces day-to-day **`serve-crop-qa`** for **end-product testing**: HugATree map chrome (dark MapLibre, glass toolbar, loader), **all** detections in memory (columnar **`.bin`** bundle, no 25k cap), **sliders + combine modes** for what counts as a tree, optional heatmap LOD at low zoom with full dataset underneath.

**Architecture (approved plan).**

| Piece | Detail |
|-------|--------|
| **Build** | `data-build-explorer-bundle` → `exports/inference/explorer/` (`detections.bin`, `register.bin`, `meta.json`); `dist_register_m` via cKDTree; optional `has_rgb` |
| **Serve** | `serve-explorer` — **`/explorer`** map + **`/api/crop/wgs84`** (extract `CropQAService` shared with Starlette) |
| **Perf** | One fetch ~13 MB; filter mask in **Web Worker**; Deck.GL **binary** scatter; heatmap when `zoom < 12` |
| **Combine modes** | v2/v3 only, AND/OR, max/min, weighted, **register_split**, disagreement highlight |
| **Filters** | BEV min, per-verifier thresholds, discoveries-only distance, require RGB |

**Ghentree path.** Same scatter record shape as HugATree; explorer is the PoC before overlaying model detections on municipal DB.

**Out of scope for v1.** Public deploy, HugATree DB/hugs, BEV retrain from verifier labels.

**Status (19 Nov):** **Shipped.** `treelidarnet/explorer/` — Flask app, binary bundle reader, combine + filter worker, static JS.

| Delivered | Detail |
|-----------|--------|
| **CLI / Make** | `data-build-explorer-bundle`, `serve-explorer`, **`make explorer-serve`** |
| **UX** | Dark MapLibre (Carto Dark Matter); Deck **9.2.7** binary attributes; **Style** pop-out (size/color sliders, representations, color-by field); min–max from **visible (filtered)** detections only |
| **Combine** | Shared `CombineConfig` (either_or, both_and, v2_only, v3_only, weighted, register_split, disagreement, …) |
| **3D dock** | Shared **`crop_viewer_3d.js`** — **Z-up OrbitControls** (map pan/zoom unchanged) |
| **Rescore** | Optional **`--rescore-on-click`** loads both checkpoints; default **export scores only** |

**Transcript / ops notes:** Port **8765** shared with crop QA — start explorer only after **`viewer-stop`**. Browser hard-refresh after static/template changes.

---

### 46. Explorer manual false-positive flags (verifier retrain) (19 Nov)

**Problem.** Hardy **building-related** and other FPs survive **BEV ≥ 0.85** and often **high v3** — GRB hard negatives sample mask cells, not necessarily the **exact detection peak**. End-product QA needs **surgical negatives** at real failure modes without carpet-style BEV relabeling (§39).

**Scope (v1).** **Verifier-only:** flag wrong peaks as **`label=0`**, `source=explorer_fp`. No BEV retrain, no FN (“missing tree”) positives, no removing peaks from shipped gzip without re-export.

**Persist**

| Piece | Path / behaviour |
|-------|------------------|
| Manifest | `data/manifests/explorer_qa_flags.parquet` |
| Dedup | `flag_id = fp_{sha256(round(lon,7), round(lat,7))[:12]}` |
| Soft delete | `active=false` on unflag |
| Audit columns | bev, v2, v3, dist_register_m, reason, note, flagged_at |

**API** (`treelidarnet/explorer/app.py`): `GET/POST /explorer/api/flags`, `DELETE /explorer/api/flags/<id>`, optional `POST .../export` snapshot.

**UI** (`explorer_flags.js`, dock): **Mark false positive**, **Unflag**, reason dropdown, note, **`f`** shortcut; **Hide flagged** filter + magenta flag layer on map.

**Training merge** (`build_crop_index --qa-flags`): active flags → same chip/crop pipeline as register rows; skip missing chip or &lt; min points; `split = _split_bucket(flag_id)`.

**Workflow** ([`docs/DEV.md`](docs/DEV.md)): flag in explorer → `data-build-crop-index` → `train-crop` (v2 + v3 configs) → **`filter-export-verifier-full`** → `explorer-bundle`.

**Scale expectations (19 Nov discussion).**

- **Manifest dilution:** ~70k+ crop rows; **dozens** of flags are **high value** but **low frequency** per epoch — target **failure-mode diversity** (flat roof, wall corner, ridge, shed), not hundreds on one street.
- **Product vs metric:** **~30–80** careful building FPs can justify a **retrain + re-export** for **local** product cleanup; global val F1 may barely move.
- **Export rule (updated §47):** Dual **`filter-export --live`** **keeps** on **`0.5·v2 + 0.5·v3 ≥ 0.45`** by default (`--combine-mode weighted`, metadata on gz). Explorer combine modes beyond weighted (e.g. **register_split**, **either_or**) remain **client-side** until promoted to CLI.
- **Training tweak (deferred):** First retrain **unchanged** BCE + `balance_class_loss`; if flagged sites stay hot, add **~10–20×** sampling or per-row loss weight on **`explorer_fp` only** (not global negative crank).

**Tests:** `test_explorer_qa_flags`, `test_build_crop_index_qa`, flags routes in `test_explorer_app`.

**Still out of scope:** Auto-retrain on flag; rebuild `detections.bin` to drop flagged peaks without new export; deep-scan verifier grid over bbox.

**Post-§46 session (same day, transcript):** **`explorer_fp`** weighted fine-tune (**114** manifest rows, **~80/114** fp_v3 &lt; 0.5 on rescore); **`--rescore-on-click`** on explorer for FP QA; user observation that **v3 can nudge some true positives up** while v2 remains discovery safety net — motivates **blend** over v3-only export.

---

### 47. Weighted dual export gate (19 Nov PM)

**Context (transcript).** Explorer treats **v2 + v3** as complementary: v2 (6 ch) on geometry/intensity, v3 (9 ch) on **per-point** DHMV RGB (not ortho imagery). User: combine is **essential** to the product; full-city **filter-export** had been **v3 ≥ 0.5 only** while the map already offered weighted / OR / register_split — a deliberate gap (§46).

**Decision.** Ship **weighted blend** on dual export: **`combo = w_v2·v2 + w_v3·v3`**, default **`w_v2 = w_v3 = 0.5`**, keep if **`combo ≥ 0.45`** (was **v3 ≥ 0.5**). Rationale: slightly **more recall** at export; **end-product** can still tighten in explorer (higher combo-t, AND, discoveries-only, etc.) without re-scoring the city.

**Code.** Shared `export_keep_passes` / `weighted_combo` in `treelidarnet/explorer/combine.py`; `filter_export_live` applies on dual keep; gzip carries `verifier_combine_mode`, `verifier_combo_threshold`, weights; kept rows set **`verifier_scores.combo`** and **`verifier_score = combo`**. CLI: `--combine-mode`, `--w-v2`, `--w-v3`; default **`--verifier-threshold 0.45`**. **`Makefile`** `filter-export-verifier-full` passes weighted + 0.45.

**Scores in explorer (clarification).** **bev / v2 / v3** in `detections.bin` are **stored as computed** (0–1). Dock bars and combine filters use **raw** values. Map **colour ramp** alone uses **min–max over currently visible** detections (§45 Style).

**Ops.** Existing **`gent_city_gent_canopy_v2_verified_v2_v3_full.json.gz`** was built under **v3 @ 0.5** until re-run. After change: **`make filter-export-verifier-full`** then **`make explorer-bundle`**. Optional future: promote **`register_split`** or FP-tuned **`_fp`** checkpoints into the same Makefile paths.

**Not changed:** BEV extract **0.80** on infer; verifier scoring pool still **BEV ≥ 0.85** on full filter job; single-checkpoint **`filter-export`** still threshold on one head unless dual ckpts given.

---

### 48. Post–re-export FP iteration (frozen BEV) (19 Nov)

**Assumption.** **BEV stays frozen** (`epoch3_hot`, **`gent_city_gent_canopy_v2.json.gz`** @ **0.80** unchanged). All product iteration is **verifier + export gate + explorer combine** on the **same detection peaks**. You only re-run **`infer-gent`** if the heatmap checkpoint changes (not planned).

**Two phases**

| Phase | What | Outputs |
|-------|------|---------|
| **A — Baseline product** | Full dual score + **weighted keep @ 0.45** on BEV≥0.85 pool | `…_verified_v2_v3_full.json.gz` → **`explorer-bundle`** |
| **B — FP loop** | Flag in explorer → index → fine-tune v2/v3 → **filter-export again** (same BEV gz in) → new bundle | Updated **v2/v3** on disk; map reflects new scores after bundle rebuild |

**When to flag more FPs (after phase A).** Use the **re-exported** map as truth for QA. Add **new** failure modes (diverse buildings, ridges, glass, sheds) — not dozens on one roof. Old flags stay in `explorer_qa_flags.parquet` unless **Unflag**; `data-build-crop-index` merges **all active** flags each time.

**Retrain strategy (second and later passes)**

| Situation | Suggested approach |
|-----------|-------------------|
| **+20–50 new flags**, prior `_fp` run helped | **`resume_from`** → **`runs/…_fp/best.pt`** (or latest `*_fp2*`); **5–8 epochs**, optional **lr `2e-4`** override on CLI if added; keep **loss×15 / sample×10** until flag count is large |
| **Major refresh** (~100+ new flags, or drift on discoveries) | **`resume_from`** → **pre-FP baseline** `gent_crop_verifier_v2/best.pt` / `gent_crop_verifier_v3_rgb/best.pt`; **15 epochs @ 0.0005** (configs default) — relearns **all** flags from a clean verifier |
| **Same 15 epochs again every small batch** | **OK but wasteful** — val F1 gate limits damage; risk **slow drift** on discoveries / marginal trees; prefer **shorter incremental** runs |

**Always between verifier trains**

1. **`eval-explorer-flags`** → `exports/qa/explorer_fp_rescore.parquet` (baseline vs new ckpt on flagged sites).
2. Explorer **`serve-explorer --rescore-on-click`** with **`_*_fp/best.pt`** — spot **discoveries** and register trees, not only flagged pins.
3. **`best.pt` by val F1** — if train run looks worse, keep previous `_fp` checkpoint for export.

**Re-export (phase B tail)**

- Input: **unchanged** `exports/inference/gent_city_gent_canopy_v2.json.gz`.
- Verifier ckpts: promoted `*_fp` (or new `*_fp2`) weights; Makefile **`filter-export-verifier-full`** (**weighted 0.45**).
- **`make explorer-bundle`** on the new verified gz — no BEV re-infer.
- Optional: point Makefile **`serve-explorer --rescore-on-click`** at `_fp` ckpts until you promote them to default paths.

**Do not**

- Re-infer the city for FP fixes alone.
- Expect global val F1 to move much — judge **flag rescoredown** + manual discovery QA.
- Drop **weighted blend** for export without checking discovery retention (v3-only was harsher).

**Naming.** New runs: `--run-name gent_crop_verifier_v2_fp2` (etc.) so **`epoch3_hot`** and baseline verifier dirs stay untouched.

---

### 49. Batched dual verifier export (19 Nov night)

**Problem.** Full **`filter-export --live`** on **~530k** peaks was **multi-day**: legacy path scored **one detection at a time**, **two full passes** (v2 then v3), **`np.load` per peak**, batch size 48 only paced **live JSON flushes** — not GPU tensor batching.

**Fix (shipped).** `treelidarnet/verifier_batch.py`:

| Piece | Behaviour |
|-------|-----------|
| **Chip grouping** | Sort peaks by **`chip_id`**; **one** `.npz` read per 50 m tile per pass over that chip’s peaks |
| **Dual crops** | Same LAZ arrays; **v2** and **v3** crops per peak with **legacy seeds** (`hash(x,y,gi)` and `hash(x,y,gi+1)`) |
| **GPU batch** | Stack crops → `PointNet++` forward in chunks of **`--infer-batch-size`** (default **64**) for v2 and v3 |
| **Order / output** | Scores written back in **input detection order**; combine + gzip unchanged (§47) |

**Resume.** `filter_export_resume.py` + default **`resume=True`**: reads **`exports/inference/live/verified_status.json`** (`scored`) and **`verified_predictions.json`** prefix, continues from **`preds[resume_done:]`**. CLI **`--no-resume`** for clean rerun. **Ops:** **stop** the old process before restart; if a new job overwrites status before resume loads, it may restart from 0 (fast path still OK).

**CLI.** `filter-export --infer-batch-size`, `--no-resume`; live flush every **512** scores (was 96).

**In progress (night log).** Re-export with **`gent_crop_verifier_*_fp/best.pt`**, **weighted 0.45**, log **`data/logs/filter_export_verifier_fp_full.log`** — batched job restarted after killing ~**33k**/530k serial run.

**Not in v1.** Multiprocess crop workers; v2/v3 fused into one forward (still two heads). BEV **`infer-gent`** already uses **`DataLoader` workers** (§31) — separate bottleneck.

---

## Decision log

| When | Decision | Rationale |
|------|----------|-----------|
| Morning | TensorBoard + periodic checkpoints | Your crash/stop risk; standard monitoring |
| Morning | Scatter inference map, not live heatmap | Performance and clarity vs HugATree |
| Midday | **Canopy v2 task** | Register incomplete; product = citywide LiDAR candidates |
| Midday | Unknown mask + audit-primary eval | Avoid punishing unlisted canopy; honest metrics |
| Afternoon | GRB buildings + roads (2 m buffer) | Hard FPs on roofs/roads without random park negatives |
| Afternoon | Rebuild chips + **fresh** v2 train from v1 weights | GRB must be in every chip; resume would mix old loss state |
| Afternoon | Defer full-grid chip build | Laptop RAM / display stability |
| Afternoon | DataLoader parallelism mid-run | GPU starved on data load; safe with resume |
| Evening | Stop train for full live infer | Demo priority; GPU contention |
| Evening | Fix live flush + status polling | Correct progress UX |
| Evening | Push `5d48c47` | Remote was behind; code-only artifact |
| Late evening | Export floor **0.20**, slider on **ghent.html** | More candidates without retrain; user filters upward |
| Late evening | Re-infer with live export at 0.20 | Single city bundle + polling map aligned |
| Late evening | Pause full grid; **patch heatmaps** in place | Variable σ without LAZ rebuild; avoid concurrent write races |
| Late evening | Grid **`--skip-existing`** after patch | Complete ~139k origins; keep ~50k patched chips |
| Late evening | Push **`952033d`** — σ, patch CLI, grid resume | Remote matches chip pipeline |
| Late evening | Patch **done** → grid **resumed** | Sequential; overnight build |
| 11 Nov AM | Train on **133k** index; archive old `best.pt` | Full grid + variable σ; fresh optimizer |
| 11 Nov AM | **No CHM / unsupervised / wider net** mid-run | Finish the σ + grid experiment first |
| 11 Nov late morning | **`tree_chip_weight: 15`**, resume ~step 9k | ~57% of steps are register chips; infer stays full grid |
| 11 Nov midday | Confidence GIF 20↔40 from last infer | QA of the **old** checkpoint, not this run |
| 11 Nov 15:17 | Keep training through **epoch 2–3** | Epoch-1 loss 0.000394 is a baseline, not a map |
| 11 Nov PM | **Per-tree peak loss** next, not mid-run | Equal penalty per stake; no hard “all trees detected” rule |
| 11 Nov evening | Weight **0.01**; watcher snapshots epoch 2 immediately | Weight 1 would swamp the Gaussian (and variable σ); the finite-loss file only lasts ~500 steps |
| 11 Nov PM | Val F1 is a **later check**, not a train loop | Held-out audit F1 vs archive before full-city infer |
| 11 Nov ~16:25 | **Candidate network** after detection is solid | Train on a LiDAR crop of **every** eligible register tree, including misses. Run on detections. Coarse type only; register keeps species |
| 12 Nov ~01:06 | **Emphasize: all register trees, not the matched detections** | The second model’s positives are the whole eligible register. Hits alone would be the easy trees. Inference still only scores detector proposals (§36) |
| 12 Nov ~01:24 | **PointNet, then a small PointNet++** | The detector stays the BEV U-Net. The crop is a column under a crown. Plain PointNet is the verifier. Two scales, radii in meters, about **512** points, if poles and shrubs still pass. No Point Transformer |
| 11 Nov ~18:45 | **Discoveries stay allowed** under per-tree loss | Unknown mask unchanged; the new term only pulls within **2 m** of a stake |
| 11 Nov ~18:45 | Keep the discovery line at **5 m** | Audit bucket only. Match is **3 m**. Generous for street spacing, tight when the crown peak is far from the stake |
| 11 Nov ~18:45 | **Audit F1** is the infer score we read; **strict** is secondary | Both written at infer/eval and shown on the map. Training does not use either; `best.pt` stays lowest train loss |
| 11 Nov ~19:58 | Restart epoch 3 in **fp32**, grad clip **5.0**, from step **75,500** | fp16 turned a saturated score into exactly 1, then NaN at step 76,511. The thousand steps before that are not the resume point |
| 11 Nov ~20:20 | Unknown band **5 m**, not **12 m** | Front gardens were inside the negative ring. Patched the 10,686 register chips; resumed at step 76,992 with Adam kept |
| 11 Nov ~22:05 | **No change** mid-epoch | `loss/tree` still falling (median 0.044). `loss/pos` rising means the peak is beside the stake, not that the run should stop |
| 11 Nov ~22:12 | Finish **this epoch**; do not train to epoch 10 for a new best | Median `loss/tree` flattened at **0.044**. The hard tail is stuck near a mean of **0.10** |
| 11 Nov ~22:12 | **No wider U-Net** | Stages are 32/64/128. Width will not split a cluster or move a crown peak |
| 11 Nov ~22:12 | Shape stays out of this detector | Height above ground is already in the chip and unused. Later: height-summary channels, then the crop model on the points |
| 11 Nov ~22:14 | Read the new epoch loss on its own scale | Total is grid loss plus `0.01 × loss/tree`. A print near **0.001–0.002** is expected. Do not compare it with **0.000366**. Later epochs compare with each other. Audit F1 is the map check |
| 11 Nov ~22:21 | Show each detection’s **LiDAR points** on Ghentree | Same crop the second model sees. A static isometric image, or an interactive 3D view of a thinned cloud, opened from one ring. The city map stays circles |
| 11 Nov ~22:24 | Full infer is an overnight **13–14 h**; register-only is about **1 h** | About **0.37 s/chip**, mostly the Python point-to-grid step. Do not run it beside training |
| 11 Nov ~22:27 | **Infer outward from the center** of the chip grid | Live map fills the middle of Ghent first. Final detections are unchanged. Training order is unchanged |
| 11 Nov ~22:58 | **Register chips before empty chips**, each still outward | One sort from the center of the whole grid, then `tree_count > 0` in front. `--limit` is the central register chips |
| 11 Nov ~22:34 | Read `loss/train_step` as the new total | Grid focal loss plus `0.01 × loss/tree`. Near **0.0015** on this pass. The epoch print is the average of these steps |
| 11 Nov ~22:37 | Keep the **0.25 m** grid | Local maxima at ≥ 0.20 inside 2 m, then NMS. Placement error is about **12 cm**. Closer trunks are merged by the mound and the **2 m** rule |
| 11 Nov ~22:55 | **Clear the ignore ring** after this epoch, then one more epoch | The 8–15 m annulus hid ~30% of stakes and tiles do not overlap. `ignore` is zeros in code. `data-patch-chips-ignore` waits until training is not reading the chips |
| 11 Nov ~23:15 | Faster infer, **same detections** | Point-order `index_add_` matches the old cell sums. Four workers. Infer does not read the training masks. The **13–14 h** figure was the Python loop |
| 12 Nov ~00:15 | **Do not train this loss again** before a penalty on unknown peaks | 400 central chips gave **20,398** rings. Usable near **0.85**. The unknown mask let bumps far from stakes stay confident. Next: patch ignore, a light unknown-peak penalty, one epoch, then re-check the center |
| 12 Nov ~00:17 | **Get the model off hallucinated trees** | The slider only hides them. Discoveries of real unlisted crowns stay. Empty ground has to lose on the next epoch, and the center at **0.20** has to look like a tree map |
| 12 Nov ~00:21 | **Keep the high-confidence trees** while punishing the carpet | The grid is the **0.50–0.85** frame, **2–5 m** inside chip corners. From **0.90** that frame matches the area. The penalty must not be “far from a stake.” Trees already scored **≥ 0.90** are the set that has to survive |
| 12 Nov ~00:15 | Live slider opens at **0.85**; rings draw **over** the green | Export floor stays **0.20**. The slider is how to look at a hot checkpoint. Amber on top so a detection is visible on its register tree |
| 12 Nov ~00:36 | **Fetch the other flight line tomorrow** | **1,841** register chips have no points. The center query saved one strip per **500 m** square. The edge strip (same corner, other flight id) was never downloaded. Rebuild those chips before the next epoch, and read every file for that corner |
| 12 Nov morning | **Second strip downloaded; penalize mid-score unlabeled peaks** | **1,864** new LAZ files. Index keeps every file per corner. Loss term is the mean of unlabeled local maxima in **[0.20, 0.90)**, weight **0.1**, **2 m** window. Peaks at **≥ 0.90** are outside it. One epoch from `best.pt` after the empty-chip rebuild and the ignore patch |
| 12 Nov ~10:50 | **epoch4 flat map** | weight **0.1** collapsed scores to ~**0.20**; restore **`epoch3_hot`** |
| 12 Nov ~11:10 | **epoch4b anticarpet + QA** | weight **0.01**, band **0.45–0.88**; center QA failed high-conf recall; revert **`best.pt`** |
| 12 Nov ~11:20 | **Freeze BEV; crop verifier** | Carpet acceptable at **0.85** view; do **not** label carpet/discovery peaks as verifier negatives (§39) |
| 16 Nov | **PointNet++ verifier first** | Skip plain PointNet staging; two-scale SA on ~512-pt crops ([docs/SPEC.md](docs/SPEC.md) §6.5) |
| 16 Nov | **Full register on crop map** | HugATree one-shot load; no bbox refetch |
| 16 Nov | **Verifier val must include negatives** | Hash-split GRB rows; val ~1.5k neg |
| 16 Nov | **Skip empty crops in train** | No zero-placeholder tensors with real labels |
| 16 Nov | **Balanced BCE + val F1 checkpoint** | Filter goal is specificity, not lowest BCE |
| 16 Nov | **`gent_crop_verifier_v2` run** | Do not trust v1 val loss; rebuild index first |
| 16 Nov | **RGB on verifier only** | DHMV has RGB; BEV stays frozen; patch npz + 9-ch PointNet++ (`v3_rgb`) |
| 16 Nov PM | **Full city infer @ 0.80** (no live) | 0.20 live OOM; ~1.17M preds shippable; default export promoted |
| 16 Nov PM | **`--resume-live` on infer-gent** | Never restart full live from chip 0 without backing up `live/` |
| 16 Nov PM | **Skip `.copc.laz` in LAZ index** | RGB patch crashed on invalid COPC sidecars; real `.laz` siblings kept |
| 16 Nov PM | **RGB patch > full live verifier** | v3 color training before scoring 1.17M detections |
| 16 Nov ~19:30 | **Pause full-grid RGB; `--register-chips` first** | v3 must not drop ~20% register positives; **2,166** tiles before train |
| 16 Nov PM | **v3_rgb `best.pt`** by val F1 @ 0.5 | Epoch **14**, F1 **~0.899**; `latest` higher precision, lower recall |
| 16–17 Nov | **Dual verifier on crop QA** | Live rows carry **`verifier_scores {v2,v3}`**; compare UI + halos |
| 17 Nov | **Full-grid RGB done** | **79,460** writes; **`make patch-chips-rgb-full`** |
| 17 Nov | **`filter-export-verifier-full`** not `infer-gent` | **530,258** @ **BEV ≥ 0.85**; dual score; BEV gz unchanged if infer aborted mid-run |
| 17 Nov | **Discoveries-only map filter** | Hide preds near register; **2 m** default slider |
| 17 Nov | **BEV ← verifier pseudo-labels** | Deferred; use dual infer rules first (§44) |
| 17 Nov | **Flask QA explorer (§45)** | Unified HugATree-style map after full dual verify; columnar bundle + combine presets |
| 19 Nov | **Explorer shipped (§45)** | Precomputed dual scores; Style + combine; **`make explorer-serve`**; no GPU on map by default |
| 19 Nov | **Manual FP → `explorer_fp` (§46)** | Third negative source beside register + GRB; verifier retrain loop, not export-only flags |
| 19 Nov | **Building FPs: flag + retrain first** | GRB ≠ peak geometry; optional stricter v3 threshold / future export combine; **no** FP-only finetune |
| 19 Nov | **`explorer_fp` loss upweight** | **Deferred** — vanilla retrain once, then 10–20× sampler if needed |
| 19 Nov PM | **§47 weighted export @ 0.45** | Align prod keep with dual-verifier story; looser than v3-only 0.5; explorer stricter filters stay QA |
| 19 Nov PM | **§48 frozen-BEV FP loop** | Full weighted re-export first; later flags → incremental `_fp` train + re-`filter-export`, no `infer-gent` |
| 19 Nov night | **§49 batched filter-export** | Chip-grouped + GPU batch; resume live JSON; same scores semantics as serial path |

---

## Results snapshot (session)

| Item | Value / note |
|------|----------------|
| LAZ | Gent bbox DHMV II under `data/raw/laz/`. **1,864** extra files so a corner can have more than one flight line (§38). Empty chips are being rebuilt |
| Chips (train) | ~10 686 register-skewed `.npz` |
| Labels | ~59k eligible register points in chips |
| v1 run | `runs/gent_real_v1/best.pt` — register matcher baseline |
| v2 config | `configs/gent_canopy_v2.yaml` — 10 epochs planned |
| v2 epoch loss (10.7k, 10 Nov 2026) | ~0.000336 → ~0.000332; on-disk archive dir `pre_full_grid_2025-11-11` (name predates calendar fix) |
| v2 epoch loss (133k, 11 Nov) | Epoch 1 **0.000394**; epoch 2 **0.000366** (grid only) @ step 75,500. Epoch 3 total **0.0021** at step **110,242** (`epoch3_hot` / live `best.pt`). That total includes `0.01 × loss/tree` |
| v2 failed anti-carpet (12 Nov) | **epoch4_flat** ~**0.021** @ 143k (flat heatmap). **epoch4b** ~**0.0039** @ 155k — QA fail, reverted |
| v2 QA gate | `qa-checkpoint-center`: rings/chip + ≥0.90 stake recall on 400 center chips |
| Smoke infer | 200 chips, epoch-1 checkpoint — low strict F1, meaningful discoveries |
| Full live infer | 10 686 chips, `gent_canopy_v2/best.pt`, `ghent_live.html` |
| Git | `2c5f620` cleared the ignore ring in code and put register chips first. The faster infer (§31) is the commit after that |
| Map QA (10 Nov evening) | Large crowns weak; street trees and discoveries strong |
| Export policy | **Shipped** full city @ **0.80** extract (`gent_city_gent_canopy_v2.json.gz`, **~1.17M** preds). Center-only export archived. Slider still filters view |
| Full city infer | **133k** @ **0.80**, non-live — Done 16 Nov (`infer_full_080.log`) |
| Live infer | **0.20** full pass **OOM ~49k**; **`--resume-live`** added; avoid 0.20 + live at city scale |
| Second live pass | Re-infer 10 686 with `--live` + 0.20 floor (`infer_conf02_live.log`) |
| Heatmap patch | **50 451** chips; GRB refresh | `patch_heatmap.log` — Done |
| Full grid | **133,000** index; **10,686** positive | `chips_index_full.parquet` |
| Heatmap σ | Per-tree from `stamomtrek` | In all built/patched chips |
| Confidence GIF | 20%↔40% on the 10 Nov city export: 46,760 preds at 0.20, 981 at 0.40 |
| Next loss | **Per-tree peak loss** weight **0.01**, fp32, grad clip **5.0** (§21). Then held-out **audit** F1 |
| Later model | **Candidate network** (§19, §36, §37, §39) | PointNet++ v2 done; **v3_rgb** after register RGB gate (§43) |
| Crop manifest | `crops_index.parquet` | **73.7k** rows; **58.9k** register rows — **~46.9k** loadable for v3 pre–register pass (**~11.9k** skipped) |
| Verifier run | `runs/gent_crop_verifier_v2/` | `best.pt` by val **F1** (~0.88) |
| Verifier RGB (manifest) | default `data-patch-chips-rgb` | **First walk complete** (16,843 scanned); **~80%** chips valid on disk |
| Verifier RGB (register) | `--register-chips` | **Done** — **2,165** (`patch_chip_rgb_register.log`) |
| Verifier RGB (manifest gap) | remainder manifest | **Done** — **1,155** |
| Verifier RGB (full grid) | `patch-chips-rgb-full` | **Done** — **79,460 / 83,290** |
| v3 train | `gent_crop_verifier_v3_rgb` | **`best.pt`** val F1 **~0.899** (epoch 14) |
| Dual verifier live | `filter-export --live` + v2+v3 | Pilot **25k** done; **full** @ BEV≥0.85 → **`_v2_v3_full.json.gz`** (**402,749** kept under **old v3≥0.5**; **re-export** for §47 weighted @ **0.45**) |
| Crop QA | `serve-crop-qa` :8765 | Dual scores; discoveries filter **2 m** slider |
| Explorer bundle | `exports/inference/explorer/` | **402,749** dets; **59,619** register; `detections.bin` + `register.bin` |
| Explorer FP flags | `explorer_qa_flags.parquet` | Curator QA → **`explorer_fp`** rows in `crops_index` |
| LinkedIn reel | `scripts/render_verifier_linkedin_reel.py` | MP4: map zoom + RGB splat; needs ffmpeg + verified JSON |

---

## Limitations and open threads

- **Coverage:** Full **133k** chip index for **infer**; **train** oversamples **~10.7k** register chips (§16).
- **Checkpoint:** Live **`best.pt` is frozen to `epoch3_hot`**. Promote only after **`qa-checkpoint-center`** passes (§39).
- **Loss / carpet:** Anti-carpet BEV experiments **hurt high-conf stakes** or **flattened** the map (§39). Carpet at export **0.20** is expected on epoch3; **0.85** view + **PointNet++ verifier** is the chosen path. Ignore ring cleared in train via `clear_stored_ignore`.
- **Shape:** The U-Net sees averaged cells. Height above ground is stored per point and not used. A wider net is not the next step. Height-summary channels, then the crop model (§19, §24).
- **Validation:** No held-out F1 in the train loop. After infer, the map shows audit F1 (primary) and strict F1 (secondary), frozen at the export floor.
- **Large trees / clusters:** §10 + §14 — σ helps singles; **silent clusters** are weak/missing peaks, not mainly NMS.
- **Threshold:** **Production export 0.80** on full grid; **0.20** kept for small diagnostics only. Slider on map for stricter view.
- **Multimodal:** Satellite + LiDAR raised by you; not implemented (DSM/DTM optional future).
- **Candidate model:** **v2 + v3_rgb** on explorer map (§44–§47). v3 helps **register**; v2 kinder on **discoveries**. **Building / roof FPs** often need **`explorer_fp`** at the peak (§46) — GRB alone is insufficient. Prod **`filter-export`** default keep is **weighted 0.5·v2+0.5·v3 ≥ 0.45** (§47); other explorer combine modes still **client-only** until promoted.
- **BEV retrain on verifier labels:** discussed **17 Nov** — only with conservative mining; default stay **frozen epoch3_hot** (§44).
- **RGB / season:** DHMV is **2013–2015**, leaf-off; RGB helps crown texture and QA visuals but is not summer inventory ground truth.
- **Crop QA / ops:** TensorBoard is manual; training does not spawn it. Crop map needs parquet + chips on disk.
- **Flanders-wide / OSM:** Stated as later-stage generalization and optional alignment data.
- **Ops:** QGIS + parallel heavy jobs compete with train/infer on same machine. Training **stalls** on cold chip cache; warm cache gives ~**20 ms/step** (§39).
- **Coverage holes:** Second-strip download and empty-chip rebuild **done** for training index (§38). Remaining east–west streak QA is separate from the verifier pivot.

---

## Related technical documentation

| Doc | Use in write-up |
|-----|-----------------|
| [docs/SPEC.md](docs/SPEC.md) | Problem, architecture, milestones |
| [docs/TASK_GHENT_BROAD.md](docs/TASK_GHENT_BROAD.md) | v2 task definition |
| [docs/LABELS.md](docs/LABELS.md) | Partial inventory, training alignment |
| [docs/EVAL.md](docs/EVAL.md) | Peaks, NMS, strict vs audit, discoveries |
| [docs/DEV.md](docs/DEV.md) | Commands: train, infer-gent, live viewer |
| [docs/VISUALIZATION.md](docs/VISUALIZATION.md) | Map layers and bundles |

§13–§14: levers, chips, export, NMS, silent clusters. §15–§16: empty chips, then tree-weighted sampling. §17: curves, GIF, network size, epoch timing. §18: validation F1 and **per-tree peak loss**. §19: **candidate network** on register-tree crops (after detection). §20: discoveries stay allowed; **5 m** is the audit line; audit F1 is the infer score, strict is secondary, and training uses neither. §21: epoch 3 NaN was float16 saturation; restarted in fp32. §22: unknown band **12 m → 5 m**. §23: `loss/tree` down, `loss/pos` up. §24: this epoch is enough for this loss; no wider U-Net; point shape is stored and unused. §25: keep each detection crop for **Ghentree** as an isometric still or a small interactive 3D view of the points. §26: a full infer is about **13–14 h**. §27: `infer-gent` does register chips first, then empty chips, each growing outward from the center. §28: `loss/train_step` is the batch total (grid plus `0.01 × loss/tree`). §29: peaks are local maxima on the 0.25 m heatmap; that cell size is enough to place a tree. §30: clear the 8–15 m ignore ring after this epoch, then one more epoch. §31: infer builds the same grids without the per-point Python loop. §32: epoch 3’s center is about **50 rings per chip** at 0.20 and about **5** at 0.85. §33: the model has to stop inventing trees. §34: the 50 m grid is the mid-score corner lattice; peaks at **≥ 0.90** look like trees and have to survive the penalty. §35: the east–west streaks are a second flight line the center-point download never fetched. §36: the second model trains on every eligible register tree, including the ones the detector missed. Detections are where it runs. §37: **PointNet++** crop verifier (16 Nov: skip plain PointNet). §38: second strip on disk; empty-chip rebuild; first anti-carpet term at weight **0.1** (failed). §39: freeze **`epoch3_hot`**; light anti-carpet failed QA; **PointNet++ verifier** next with **safe negatives** (not carpet as label-0). §40: crop QA viewer; verifier training hygiene (val negatives, balanced BCE, skip bad crops, F1 checkpoint); **`gent_crop_verifier_v2`**. §41: **verifier-only RGB** — patch LAZ color into crop chips; **`gent_crop_verifier_v3_rgb`**. §42: **full city @ 0.80**, infer **resume-live**, export promotion, **COPC** patch fix, **RGB before** million-detection verifier. §43: **register-first RGB**, xy **matcher fix**, register + manifest patch. §44: **dual v2/v3 QA**, **full-grid RGB done**, **530k dual filter-export**, discoveries map filter, **no BEV overwrite** on aborted infer, **verifier-as-BEV-teacher** deferred. §45: **Flask explorer** — HugATree UX, **402k** columnar bundle, combine + Style UI (**shipped** 19 Nov). §46: **`explorer_fp`** manual flags → crops_index → verifier retrain; building-peak negatives. §47: **weighted dual export** **0.5·v2+0.5·v3 ≥ 0.45**; per-point RGB semantics; raw vs visible-normalized map colours. §48: **frozen BEV** product loop — full re-export then **incremental FP** fine-tune + re-export on same detection gz. §49: **batched** dual verifier scoring + **live resume** for `filter-export-verifier-full`.

---

*Design diary through **19 Nov 2026**: BEV **epoch3_hot** frozen; **v2 + v3_rgb** verifiers trained; **full dual export** @ BEV≥0.85; **Flask explorer** live; **manual FP flags** + **`explorer_fp`** retrain loop; **export keep** → **weighted @ 0.45** (§47); **post-re-export FP iteration** (§48); **batched verifier export + resume** (§49). Checkpoints not in git. Session transcripts: explorer QA / FP flags, weighted export, incremental FP strategy, **filter-export speedup** (Nov 19). For code state, see git history on `main`.*
