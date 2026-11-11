# TreeLidarNet — thought process (design diary)

**Session:** 10–11 November 2026 (build day, then full-grid train)  
**Repo:** [TreeLidarNet](https://github.com/CattF-LDaman/TreeLidarNet) · references: HugATree, LuweDaman (Deck.GL map UX)  
**Purpose of this file:** A readable narrative for a **project write-up** — what we wanted, how the goal changed, what we built and observed. Technical truth lives in `docs/SPEC.md`, `docs/TASK_GHENT_BROAD.md`, `docs/LABELS.md`, `docs/EVAL.md`.

---

## One-page story

We set out to detect trees in Ghent from **Flanders DHMV II aerial LiDAR** (2013–2015 point clouds) and compare to Stad Gent’s **open tree register** (~68k rows, ~59k eligible for training). The first plan was essentially **“match the city inventory”** (v1: register positives, negatives everywhere else). That misaligned with reality: the register is **partial** (city-managed trees only), so punishing predictions on unlisted canopy creates false “errors.”

Mid-session we **pivoted** to *“find trees in Ghent broadly”* (canopy v2): keep register locations as **soft positives**, add an **unknown mask** so unlabeled canopy is not penalized, and add **GRB** building/road masks as **hard negatives**. We trained with PyTorch on **50 m LAZ chips**, fine-tuned from v1, ran smoke and **full live city inference** on a Deck.GL map, and pushed code commit `5d48c47`.

After map review, you asked for **more detections without retraining**: lower the **export confidence floor** from 0.35 to **0.20**, add a **confidence slider** on the full city map, and **re-infer** with live polling so rings fill in during the run.

Overnight the **full 50 m grid** finished (**133k** chips) after an in-place **variable-σ** heatmap patch. On 11 Nov we archived the pre-grid weights and started a long fine-tune. Most grid chips have **no register trees**; we **oversampled** the ~10.7k that do. Epoch 1 of that pass landed at loss **0.000394** with only a slight drop in smoothed `loss/pos`. The agreed next training change, after epochs 2–3, is a **per-tree peak loss** so each missed stake counts — not a hard rule that every register tree must become a detection, and not a bigger network yet. Later, once detections are trustworthy, a **second network** scores each candidate on a small LiDAR crop: tree or not, and only a coarse structure class. It **trains on register trees**; the detector only chooses where to run it. The register itself stays the rich species table.

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
| **NMS** | After peak pick: drop weaker detections within ~2 m — fixes duplicates, not “no peak at all.” |
| **CHM** | Canopy height model: DSM − DTM. **CHM peaks** are hilltops on that height map (non-learned baseline). |
| **Validation F1** | Audit/strict F1 on a **held-out** area, used to pick a checkpoint. Train loss is not that score. |
| **Per-tree peak loss** | Planned next loss: focal penalty per register tree (stake or max in ~2 m), averaged over trees. |
| **Candidate network** | Later second model: LiDAR crop around a point → tree or not, optional coarse type. Trained on **register** trees; run on detections. |

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
| **16:23–16:25** | Later: detections only, then a network for type / “is it a tree”? | **Second stage** on LiDAR crops. **Trains on register trees.** Species stays coarse (leaf-off LiDAR). Register remains the attribute table |

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
| `ignore` | Edge annulus — mask loss |
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

**Training the second network uses register trees.** For each eligible `gis_id`, crop the LiDAR around the stake and teach “this is a tree,” plus `sortiment` when it is a real label (not `Te Bepalen`). Those crops exist even if the detector missed the tree. Add non-trees on purpose (GRB roofs and roads, detector peaks that are not canopy). Discoveries are not type labels until someone checks them.

**At inference,** the detector proposes points. The second network scores each point. If it falls within ~3 m of a register row, **attach** that row. Unmatched points stay discoveries with LiDAR scores only. The register is not re-learned as the species database.

**Limits.** DHMV II is winter, leaf-off. Full species or cultivar from this LiDAR is not realistic. Conifer versus broadleaf might be. One detector peak for a whole cluster cannot be split into five types by the crop model.

**When.** After per-tree peak loss, held-out audit F1, and a city infer you trust. Then: detections-only export, crop dataset from register trees plus hard non-trees, **verifier first**, coarse type only if the verifier helps.

**For write-up:** The map product is a list of candidates. The register labels the ones it already knows. LiDAR classification fills in “tree?” and a rough structure for the rest.

---

## Where we left off (11 Nov, ~16:25)

Training is **in epoch 2** (`train_full_grid.log`, `tree_chip_weight: 15`). Epoch 1 of this pass is **0.000394** at step **42,250**. Pre-grid weights are archived. Infer coverage is still the full **133k** index; this run has not been inferred yet.

**Do not interrupt for a loss or architecture change.**

**After epochs 2–3:**

1. A one-shot watcher (`scripts/switch_after_epoch.py`) snapshots the epoch-2 checkpoint immediately (the file is overwritten ~500 steps later), stops the old-loss process, restores that snapshot, and restarts `--resume`. Per-tree weight is **0.01** so the Gaussian loss still counts.
2. Small held-out infer vs the archive (audit F1, large trees vs clusters).
3. Full-city `infer-gent --score-threshold 0.2` only once that checkpoint is the one we want on the map.
4. **Then** a detections-only export and a **candidate network** trained on register-tree LiDAR crops (§19): verifier first, coarse type second.

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
| 11 Nov ~16:25 | **Candidate network** after detection is solid | Crops trained on **register trees**; run on detections. Coarse type only; register keeps species |

---

## Results snapshot (session)

| Item | Value / note |
|------|----------------|
| LAZ | Gent bbox DHMV II tiles under `data/raw/laz/` |
| Chips (train) | ~10 686 register-skewed `.npz` |
| Labels | ~59k eligible register points in chips |
| v1 run | `runs/gent_real_v1/best.pt` — register matcher baseline |
| v2 config | `configs/gent_canopy_v2.yaml` — 10 epochs planned |
| v2 epoch loss (10.7k, 10 Nov) | ~0.000336 → ~0.000332; archived in `pre_full_grid_2025-11-11` |
| v2 epoch loss (133k, 11 Nov) | Epoch 1 **0.000394** @ step 42,250; tree-weighted; epoch 2 running |
| Smoke infer | 200 chips, epoch-1 checkpoint — low strict F1, meaningful discoveries |
| Full live infer | 10 686 chips, `gent_canopy_v2/best.pt`, `ghent_live.html` |
| Git | Origin `952033d` (variable σ). Sampler, GIF script, and this diary are still local |
| Map QA (10 Nov evening) | Large crowns weak; street trees and discoveries strong |
| Export policy | Min confidence **0.20** at infer; map slider from 0.20 up (`infer-gent --score-threshold 0.2`) |
| Second live pass | Re-infer 10 686 with `--live` + 0.20 floor (`infer_conf02_live.log`) |
| Heatmap patch | **50 451** chips; GRB refresh | `patch_heatmap.log` — Done |
| Full grid | **133,000** index; **10,686** positive | `chips_index_full.parquet` |
| Heatmap σ | Per-tree from `stamomtrek` | In all built/patched chips |
| Confidence GIF | 20%↔40% on the 10 Nov city export: 46,760 preds at 0.20, 981 at 0.40 |
| Next loss | **Per-tree peak loss** (§18), after epochs 2–3, then held-out audit F1 |
| Later model | **Candidate network** (§19) | Register-tree crops → tree? / coarse structure; detections-only export |

---

## Limitations and open threads

- **Coverage:** Full **133k** chip index for **infer**; **train** oversamples **~10.7k** register chips (§16).
- **Checkpoint:** Epoch 1 of the full-grid pass is only a new baseline; `best.pt` is lowest **train** loss, not validation audit F1.
- **Loss:** Grid-mean focal loss dilutes missed register trees; **per-tree peak loss** is next (§18), after epochs 2–3. Not a hard all-trees-detected constraint.
- **Validation:** No held-out F1 in the train loop; `best.pt` tracks train loss only.
- **Large trees / clusters:** §10 + §14 — σ helps singles; **silent clusters** are weak/missing peaks, not mainly NMS.
- **Threshold:** Export floor now **0.20** (infer); val tuning still future; slider on full map for QA.
- **Multimodal:** Satellite + LiDAR raised by you; not implemented (DSM/DTM optional future).
- **Candidate model:** Not started. Second stage on LiDAR crops, trained on register trees, after the detector is worth trusting (§19). Leaf-off LiDAR will not reproduce full `sortiment`.
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

§13–§14: levers, chips, export, NMS, silent clusters. §15–§16: empty chips, then tree-weighted sampling. §17: curves, GIF, network size, epoch timing. §18: validation F1 and **per-tree peak loss** (next training change). §19: **candidate network** on register-tree crops (after detection).

---

*Design diary through 11 Nov 2026, 18:12: export confidence, live map, variable σ, full grid, tree oversampling, per-tree loss at weight 0.01, and the epoch-2 watcher. For code state, see git history on `main`.*
