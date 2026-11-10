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

---

## Limitations and open threads

- **Coverage:** Train/infer chips skewed to register neighbourhoods; full 50 m grid not built on laptop.
- **Checkpoint:** Early fine-tune; smoke metrics misleading vs map QA.
- **Large trees / clusters:** Peak extractor + point targets + stake offset (see §10).
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

---

*Generated from Cursor session transcript (2026-11-10), including late-evening export-confidence and live-map thread. For code state after this narrative, see git history on `main`.*
