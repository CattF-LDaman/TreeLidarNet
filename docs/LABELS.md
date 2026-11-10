# TreeLidarNet — Labels and what we are actually learning

## The inventory is partial

**Locaties bomen Gent** lists **city-managed** trees Stad Gent tracks (~68k points). It is **not** every woody plant LiDAR can see:

- Private garden trees, many park understory trees, hedges, and woodland often have **no row**.
- LiDAR (DHMV II, 2013–2015) sees **canopy** everywhere; labels mark **register locations** only.

**Primary task (v2):** find **tree-like canopy in Gent** from DHMV II LiDAR — citywide peaks + scores ([TASK_GHENT_BROAD.md](TASK_GHENT_BROAD.md)).

**Secondary:** compare to Gent register for **audit** (discoveries OK) and **strict** benchmarks — never a replacement census ([SPEC.md](SPEC.md) non-goals).

---

## Training alignment

| Choice | Rationale |
|--------|-----------|
| **Targets** | Heatmap peaks only at **eligible** `gis_id` points in each chip |
| **No “all canopy” mask** | We do not have full-tree labels; training on every green blob would need another data source |
| **Positive chips** | Built where ≥1 eligible tree lies in the 50 m patch (`data-build-gent-chips`) |
| **Negative chips** | Optional road/roof-heavy chips (spec §5.5) — avoid random park chips where **unlabeled trees** would punish true detections |
| **Unknown mask (v2)** | No negative loss far from register peaks — unlisted canopy not punished |
| **Loss ignore band** | Chip edges masked so uncertain context does not dominate |

v1 learned register-only negatives; **v2 (canopy)** keeps register **positives** but uses an **unknown mask** so the model may fire on unlisted trees.

---

## Evaluation alignment (two views)

Always report **both**; they answer different questions.

| Mode | Question | False positives |
|------|----------|-----------------|
| **Strict** | How well do we match the **official register**? | Every unmatched prediction counts (incl. plausible unlisted trees) |
| **Audit** | How well do we match the register **without punishing discoveries**? | Only predictions **near** some register point but not matched; **discoveries** (far from any GT) reported separately |

Implementation: [EVAL.md](EVAL.md) §7; `treelidarnet/metrics.py` (`fp_audit`, `precision_audit`).

**Do not** chase strict F1 alone in wooded or residential areas — high **discovery** count may mean inventory gaps or private trees, not model failure.

---

## Interpreting errors

| Pattern | Likely meaning |
|---------|----------------|
| FN near known `gis_id` | Offset, crown vs stake, leaf-off height, or model error |
| Pred with no GT within 5 m (**discovery**) | Unlisted tree, large shrub, or FP — needs map QA |
| Pred within 3–5 m of GT but unmatched | Ambiguous / offset — tune match distance or align labels |
| FP on flat roofs / low hedges | Use CHM baseline compare; building mask (v2) |

Stratify by `eigenaar`, `plantwijze`, and park vs street when reviewing ([EVAL.md](EVAL.md) §10).

---

## Visualization copy

On maps and exports, label layers clearly:

- **Green** — Gent register (ground truth), not “all trees in the world”
- **Cyan** — model matched to `gis_id`
- **Orange** — strict false positive (includes near-miss)
- **Discovery export** — `exports/discoveries_*.geojson` for human review, not automatic upload to Stad Gent

---

## Future improvements (not v1 blockers)

- Negative chips from **GRB roads / roofs** only (fewer false negatives on unlabeled canopy in parks)
- Soft labels from **ineligible** rows (post-2015) for analysis only
- Species / `eigenaar` stratified metrics in every eval report
- Optional HugATree UI: “LiDAR suggests canopy here — not in register”

---

## Related

| Doc | Contents |
|-----|----------|
| [DATA.md](DATA.md) | Eligibility, what labels are *not* |
| [EVAL.md](EVAL.md) | Matching, discoveries bucket |
| [LEGAL.md](LEGAL.md) | Do not publish discoveries as official municipal data |
