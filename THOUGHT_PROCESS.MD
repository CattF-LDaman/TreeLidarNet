# TreeLidarNet — thought process (design diary)

**Session:** 10 November 2026 (single intensive build day)  
**Repo:** [TreeLidarNet](https://github.com/CattF-LDaman/TreeLidarNet) · references: HugATree, LuweDaman (Deck.GL map UX)  
**Purpose of this file:** A readable narrative for a **project write-up** — what we wanted, how the goal changed, what we built and observed. Technical truth lives in `docs/SPEC.md`, `docs/TASK_GHENT_BROAD.md`, `docs/LABELS.md`, `docs/EVAL.md`.

---

## One-page story

We set out to detect trees in Ghent from **Flanders DHMV II aerial LiDAR** (2013–2015 point clouds) and compare to Stad Gent’s **open tree register** (~68k rows, ~59k eligible for training). The first plan was essentially **“match the city inventory”** (v1: register positives, negatives everywhere else). That misaligned with reality: the register is **partial** (city-managed trees only), so punishing predictions on unlisted canopy creates false “errors.”

Mid-session we **pivoted** to *“find trees in Ghent broadly”* (canopy v2): keep register locations as **soft positives**, add an **unknown mask** so unlabeled canopy is not penalized, and add **GRB** building/road masks as **hard negatives**. We trained with PyTorch on **50 m LAZ chips**, fine-tuned from v1, ran smoke and **full live city inference** on a Deck.GL map, and pushed code commit `5d48c47`.

After map review, you asked for **more detections without retraining**: lower the **export confidence floor** from 0.35 to **0.20**, add a **confidence slider** on the full city map, and **re-infer** with live polling so rings fill in during the run.

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
| **Discovery** | Prediction with no register tree within ~5 m (allowed in v2; counted separately in **audit** metrics). |
| **Strict vs audit F1** | Strict punishes all unmatched preds; audit treats distant discoveries as research output, not FP. |
| **Unknown mask** | Pixels far from any register point → no negative training loss (canopy may exist without a label). |
| **GRB hard negative** | Official building footprints + roads (buffered) → extra penalty for predicting trees on built infrastructure. |
| **Fine-tune** | Start from v1 `best.pt` weights; adjust for v2 loss/masks rather than train from scratch. |
| **Confidence score** | Model output 0–1 on each peak; **export floor** is the minimum kept at infer; map **slider** can hide weaker rings without re-infer. |
| **Variable σ** | Training heatmap Gaussian width from register **`stamomtrek`** (1.0–2.5 m); addresses stake-sized targets vs wide crowns on large trees. |

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
| **22:xx** | **Patch heatmaps in place**; resume grid with `--skip-existing` | `data-patch-chips-heatmap`; docs + `tests/test_heatmap_sigma.py`; train refresh after patch + grid |

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

## Where the night ended (handoff for the next day)

By the time we closed the laptop, the **full-grid LAZ build was paused** on purpose (~40k of ~139k grid steps, ~50k `.npz` on disk). Variable σ was implemented in code; existing chips were being **updated in place** rather than rebuilt from LAZ.

**What was still running:** `data-patch-chips-heatmap` over **~50,451** chips (log: `data/logs/patch_heatmap.log`; progress every 500 lines). Defaults included **GRB refresh** from `data/raw/grb`, not heatmap-only (`--no-grb` was not used). Watch with `tail -f data/logs/patch_heatmap.log`; finish when the log shows **Done patching** or **50451/50451**.

**What we agreed not to do yet:** **Wait for the heatmap patch to finish** before starting `data-build-gent-grid` (sequential run — no parallel patch + grid). Then resume grid with **`--skip-existing`** so ~50k patched chips are kept and only **missing** cells are built (new builds already use variable σ in `build_chip`). Do **not** expect the current `gent_canopy_v2/best.pt` to improve on large crowns without **retraining** — export **0.20** and the map slider only change which peaks we keep at infer time; they do not change the old **fixed 1.0 m** targets the model already learned.

**Remember tomorrow, in order:**

1. Confirm the **patch** finished (`Done patching` or **50451/50451** in `patch_heatmap.log`; no `data-patch-chips-heatmap` process).
2. **Resume the full grid** (patched chips unchanged; new cells get variable σ at build):

   `uv run treelidarnet data-build-gent-grid --laz-dir data/raw/laz --skip-existing 2>&1 | tee -a data/logs/build_gent_grid.log`

   Let this run when the machine has headroom (long job; don’t stack train + grid on a tight laptop if avoidable).
3. When ready for **citywide training** (ideally after most of the grid exists, or accept a smaller interim set): **fine-tune v2** on the patched chips — fresh run from `gent_canopy_v2/best.pt` or continued epochs with `configs/gent_canopy_v2.yaml`, not “resume” as if targets were unchanged.
4. After a new checkpoint: **`infer-gent`** again (`--score-threshold 0.2` if keeping the same export policy) and map QA on large trees / clusters.
5. Optional housekeeping: commit local docs + σ/patch CLI when satisfied; push when `main` should match the machine.

**Shortcut if impatient:** Fine-tune on the ~50k patched chips **while** the grid keeps building — valid for testing σ; full-Gent training still waits on the grid + index manifest.

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

---

## Results snapshot (session)

| Item | Value / note |
|------|----------------|
| LAZ | Gent bbox DHMV II tiles under `data/raw/laz/` |
| Chips (train) | ~10 686 register-skewed `.npz` |
| Labels | ~59k eligible register points in chips |
| v1 run | `runs/gent_real_v1/best.pt` — register matcher baseline |
| v2 config | `configs/gent_canopy_v2.yaml` — 10 epochs planned, **~3 completed** in session |
| v2 epoch loss | ~0.000336 → ~0.000332 (best epoch loss) |
| Smoke infer | 200 chips, epoch-1 checkpoint — low strict F1, meaningful discoveries |
| Full live infer | 10 686 chips, `gent_canopy_v2/best.pt`, `ghent_live.html` |
| Git | `main` @ `5d48c47` — canopy v2 code, GRB, viewers, tests |
| Map QA (evening) | Large crowns weak; street trees + discoveries strong | Informs threshold/export work |
| Export policy | Min confidence **0.20** at infer; map slider from 0.20 upward | `infer-gent --score-threshold 0.2` |
| Second live pass | Re-infer 10 686 with `--live` + 0.20 floor | `exports/inference/live/`, log `infer_conf02_live.log` |
| Full grid (partial) | ~50k `.npz`, grid build paused mid-run | Resume with `--skip-existing`; index incomplete until grid done |
| Heatmap σ | Per-tree from `stamomtrek` | Patch CLI + tests; then fine-tune v2 |

---

## Limitations and open threads

- **Coverage:** Register-skewed set still used for train until full grid completes; ~50k grid chips on disk; see **Where the night ended** for patch → grid → train order.
- **Checkpoint:** Early fine-tune; smoke metrics misleading vs map QA.
- **Large trees / clusters:** Peak extractor + point targets + stake offset (see §10); **variable σ** targets mitigate but need re-train.
- **Threshold:** Export floor now **0.20** (infer); val tuning still future; slider on full map for QA.
- **THOUGHT_PROCESS / uncommitted viewer tweaks:** May lag `main` until next push.
- **Multimodal:** Satellite + LiDAR raised by you; not implemented (DSM/DTM optional future).
- **Flanders-wide / OSM:** Stated as later-stage generalization and optional alignment data.
- **Ops:** QGIS + parallel heavy jobs compete with train/infer on same machine.

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

§13 is the **beginner-facing** summary of detection levers (labels vs train vs infer).

---

*Generated from Cursor session transcript (2026-11-10), including export-confidence, live-map, full-grid, variable-heatmap-σ, and detection-levers explainer. For code state after this narrative, see git history on `main`.*
