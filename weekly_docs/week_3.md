# Week 3 — Sep 7 (Sun) → Sep 13 (Sat)

**Objectives:** build XI + 7 subs decision layer with compatibility matrices & MF.

**Tasks:**

1. **Compatibility matrix (Sep 7–8):**

   * Construct **co‑play matrices** (minutes together), **synergy scores** (on‑pitch goal difference/xG delta while co‑playing), role adjacency weights.
   * **Matrix Factorization:** implicit‑feedback ALS or BPR to derive player latent vectors; cosine similarity = compatibility prior.
2. **Stage‑2 model (Sep 9):** gradient boosting using inputs: Stage‑1 posteriors, compatibility features per candidate XI, opponent context.
3. **XI & bench optimizer (Sep 10–11):**

   * Mixed‑integer programming (PuLP/OR‑Tools):

     * Constraints: formation role counts, fitness thresholds, min GK/DEF/MID/FWD on bench, minutes caps, injury/ban, home/away.
     * Objective: maximize **composite lineup score** predicted by Stage‑2 + synergy regularizer.
4. **Validation (Sep 12):**

   * Back‑test against historical selections & out‑of‑sample matches; measure lineup‑level metrics.
5. **G3 Stage‑2 Ready (Sep 13):** API returns XI+7 with rationale (top features/compatibility edges).

**Outputs:** Compatibility notebook, `compat_matrix.npz`, `stage2_model.pkl`, optimizer module, validation report.
