# Changelog

Major updates to the research code, results and paper, newest first within each day.

### 2026-10-29: Major update 1 (audit and redesign)

- Full audit of code, dataset and paper → [AUDIT.md](AUDIT.md).
- Physics fixes:
  - antenna gains, RIS element gain, and a facade-mounted RIS (the UAV
    was behind or edge-on to the panel before);
  - obliquity on both hops;
  - noise from kTB·NF;
  - a vectorised channel with cached ITU absorption.
- New `hardware.py`:
  - the PA moved to the BS (a passive RIS has no PA), modelled with the
    Bussgang SNDR;
  - RIS quantization, phase noise and amplitude–phase coupling;
  - S11 mismatch;
  - a hardware-aware discrete phase search.
- New metrics: SNDR, BER (QPSK/16-QAM), spectral efficiency, link
  availability, delivered array gain, and channel-estimation NMSE
  (naive LS vs hardware-aware LMMSE).
- Beam prediction redesigned: it predicts the next beam (the old target
  leaked through argmax), splits by sequence, and reports top-k, ±1 beam
  and power ratio.
- Tracking redesigned: predictors are scored by delivered array gain at
  3 horizons, all on the same noisy measurements.
- Housekeeping:
  - legacy scripts moved to `legacy/`;
  - the 286 MB `.pkl` is no longer tracked;
  - `requirements.txt` completed;
  - `.gitignore` and regression tests added.

### 2026-10-29: Major update 2 (sequence model, closed loop, GPS-ready)

- PyTorch (CPU) installed. New `beam_sequence_model.py`: a GRU with a
  residual over the last sweep and a soft power-based target. It is the
  best model at both horizons, most clearly at t+3.
- `public_data_adapter.load_scenario23_kinematics` reads UAV GPS,
  altitude, speeds, pitch and roll from `unit2/`. It converts them to
  BS-relative position, bearing and elevation. Both beam scripts use
  these features automatically once the full dataset is present. A test
  uses a fake `unit2/` folder.
- Designed trajectories moved in front of the facade-mounted RIS, so
  they are now 100% served. Tracking was re-run.
- Closed loop added to `run_physical_layer.py`: estimate the channel,
  set phases from the estimate, then measure SNDR.
- Tried the full Scenario 23 download: deepsense6g.net now requires an
  account sign-in, and no public mirror exists. This step needs a person
  (see below).

### 2026-10-29: Major update 3 (parametric channel estimation)

- New `parametric_estimation.py`: ML direction refinement from M random
  RIS pilot patterns around a GPS prior, vectorised over candidates.
- `run_physical_layer.py` adds `parametric_study`, which compares GPS-only,
  parametric and per-element LMMSE control by pilot budget.
- New regression test: the parametric estimate gets within 0.5 dB of
  perfect CSI at 50 m.

### 2026-10-29: Major update 4 (noise-aware beam metrics, adaptive estimation)

- Beam metrics now include accuracy within the measured noise floor
  (0.25 dB) and a class-balanced version. Exact top-1 and macro-F1 are
  still reported.
- `adaptive_pilot_plan` / `estimate_from_prior`: the pilot budget comes
  from the prior's angular uncertainty and the predicted SNR. The margin
  (128) was chosen on separate seeds from the reported runs.

### 2026-10-30: Major update 18 (proofread, stack-vs-stack test)

- `significance_tests.py`: new "stacking" family (stacks vs the GRU ensemble
  and all-sensing vs power-only stack). Sensing adds +1.1 / +3.4 points of
  exact top-1 inside the stack (Holm p = 0.006 / 0.045); Table XIII.
- Camera-embedding runs tried in the stack: exact top-1 moves by <= 0.1
  points, so the stack keeps four runs (stated in the paper).
- Paper proofread: Table XI refreshed from the metre/gated rerun, Sec. V-B
  describes the stacked predictor, LightGBM reference [25] added and
  checked, wording fixes (near-tie claim, Table IX caption).

### 2026-10-30: Major update 17 (stacked blind predictor, all-class macro-F1 ≥ 0.95 in closed loop)

- New `blind_stacking.py`: nested stack of the GRU variants of every
  side-information run, extra trees / LightGBM on long-history features,
  the last sweep and a shifted-template tracker (sub-beam peak, aligned
  average, predicted shift). Blind exact top-1 72.1 → 75.6% at t+1 (96% of
  the noise bound) and 56.7 → 64.3% at t+3; within noise 97.2 / 89.6%;
  per flight +3.6 / +7.6 points, Holm p ≤ 0.003; sensing adds +1.1 / +3.4
  points inside the stack (Holm p = 0.006 / 0.045). New Fig. 5.
- `beam_training_overhead.py` shortlists from the stacked scores when they
  are present: t+1 single round 99.3% / macro-F1 0.961 with 6.4 beams
  (0.50 ms); t+3 two-round 99.4% / 0.935.
- Closed loop: camera + GPS top-6 candidates give 99.46% exact top-1 and
  all-class macro-F1 0.972 at 0.66 ms; top-4 + 6 tie partners 99.57% and
  0.963 / 0.992, better on 27 flights, worse on none.
- Rerun with heights in metres and the gated locator: Tables XI–XIV,
  Figs. 9–11, stale-GPS (up to 1.1 s) and timing sensitivity in the paper.
- Limitations rewritten item by item, with what was done about each.

### 2026-10-30: Major update 16 (blind exact prediction explained, figures, docs)

- Table VII: noise bound row, exact top-1 as % of the bound, exact top-1 on
  decisive samples (GRU ensemble 94.1% at t+1) vs near-ties, fused GRU +
  last-sweep row (+0.3 / +0.6 points).
- New Fig. 1 (system overview); Fig. 3 adds the 140 GHz NF panel; Fig. 4
  adds the ridge predictor; figures renumbered.
- Sensitivity to PA back-off, obliquity exponent and D-band power.
- UAV height/altitude read as feet (take-off 1175.8 ft = local ground);
  locator outlier gate; stale-GPS and latency-model sensitivity (results
  pending the rerun).

### 2026-10-30: Major update 15 (references, NF sensitivity, closed-loop macro-F1)

- References [20]-[22] verified; no [VERIFY] items left. [21] now cites a
  measured D-band SiGe LNA (NF 4.8-6.1 dB), and the model's 10 dB is
  described as a conservative receiver NF with a 6/7/10 dB sensitivity:
  the 140 GHz RIS needs 24-32 cm.
- Closed-loop macro-F1 added: camera + GPS tracker 0.980 (>= 5 samples).
- Locator validated against GPS direction; ~1% of frames misdetected.

### 2026-10-30: Major update 14 (UAV locator, sensing-aided tracking)

- Label-free UAV locator in the BS camera (`camera_beam_prediction.py
  --locate`): camera-only within-noise 67% -> 81% at t+0, 54 ms/frame.
- Camera + GPS candidates in the closed loop: 99.2% exact top-1 at
  0.56 ms (tracker: 97.2% at 1.05 ms); paper Table XIV, Fig. 9.
- Blind prediction with the UAV pixel: no significant gain (Table XI).
- README: removed a duplicated table header and a misplaced paragraph.

### 2026-10-30: Major update 13 (camera, GPS-aided beam training, significance)

- Camera: frozen ResNet-18 embeddings; camera only and as side information
  (paper Tables XI-XII, Fig. 7). Negative result: camera features lower
  blind accuracy significantly.
- Beam training and conformal studies rerun on the GPS-aided GRU ensemble
  (`results/gps/`): small changes, within CIs.
- Per-flight significance tests (paper Table XIII, Fig. 8). The GPS gain
  at t+3 is not significant per flight.
- Fixes: Table IX covers 33 flights (flights 22 and 30 hold one sweep);
  GRU-ensemble t+1 macro-F1 is 0.48, not 0.49; Table XI balanced 89.3,
  not 89.2 (Table XI is now generated from fold means).

### 2026-10-30: Major update 12 (full Scenario 23: GPS and kinematics)

- Full download wired in (git-ignored). Raw mmWave files match the beam
  CSV (max difference 3e-8).
- GPS adds 1.1–4.2 pooled within-noise points at t+3, nothing at t+1 (paper
  Table XI, Fig. 7); per flight the GRU-ensemble gain is not significant
  (+1.1 points, 95% CI -0.1 to +2.3, Wilcoxon p = 0.34). `SCENARIO23_FEATURES=power` keeps the power-only
  pipeline reproducible.

### 2026-10-30: Major update 11 (learned models in the loop, paper v3 figures)

- `closed_loop_learned_tracking.py`: ten models trained on measured-only
  observations. None beats the tie graph at equal budget (negative result).
- Paper: Fig. 5 (conformal), Fig. 6 and Table X (closed loop, learned vs
  tie graph); tables no longer split across pages; references [2], [7],
  [23], [24] confirmed.

### 2026-10-30: Major update 10 (conformal sets, closed loop)

- `conformal_beam_training.py`: conformal shortlists with a target exact
  top-1. 95% target: 96.1% at t+1 (3 beams), 95.5% at t+3 (5.4 beams).
- Audit finding H6: reduced beam training was scored open-loop (it
  assumed full-sweep history).
- `closed_loop_beam_tracking.py`: flight replay with measured-only
  knowledge, effective spectral efficiency and latency. Tie-graph
  tracking: 97.2% exact top-1, 99.2% within noise, 1.05 ms.

### 2026-10-30: Major update 9 (95% target, beam-training latency)

- `beam_training_overhead.py`: target raised to 95%. New learned tie
  graph, two-round search with beam-loss fallback, single-burst
  graph-expanded shortlist, and a 5G NR FR2 latency model. Operating
  points minimise latency. The tie graph is fitted out-of-fold, including
  inside the nested selection (an earlier in-sample version overstated
  t+3 macro-F1 by ~0.02).
- t+1: 99.1–99.3% exact top-1, 0.955–0.961 all-class macro-F1, about
  0.5 ms of beam training. t+3: 98.9–99.4% exact top-1, 0.97–0.98
  macro-F1 (≥ 5 samples), 0.91–0.93 all-class (singleton-limited).
- Measured GRU inference latency at batch 1 (CPU): 0.5 ms single model,
  1.4 ms ensemble. Off the critical path.
- `tests/test_beam_training.py`: 4 tests for the tie graph, second
  round, fallback and burst ranking.

### 2026-10-29: Major update 8 (IEEE Word version)

- `paper/revised_paper_ieee.docx`: the manuscript in IEEE conference
  layout (US Letter, Times, two columns, 6 pages). Rebuild it with
  `python paper/build_docx.py` whenever `revised_paper.md` changes.
- Tables renumbered I–VIII in order of appearance (the old I-a / I-b
  are now II / III, and the old II–VI are IV–VIII). Sec. VI-A was
  reordered to match.
- [VERIFY] items are highlighted in yellow in the Word file.

### 2026-10-29: Major update 7 (hardware sensitivity and sub-THz)

- `phase_resolution_study`: coupling-aware control gains 0.55, 0.92 and
  1.01 dB at 2, 3 and 4 bits, and nothing at 1 bit.
- `coupling_sensitivity`: `ris_coupling_beta_min` is now a config field.
  The total hardware penalty spans 2.9–7.8 dB over the unmeasured
  coupling strength.
- `carrier_study` and D-band sizing: at equal 8 cm aperture, 140 GHz
  slightly beats 60 GHz on propagation. With published D-band hardware
  (14 dBm average, NF 10 dB) it needs a 32 cm RIS for full availability,
  vs 24 cm at 60 GHz.
- Paper updated: coupling and D-band tables (now Tables II and III), the 140 GHz paragraph, and
  references [20] and [21] (marked to verify).
- Beam GRU: beam-shift/mirror augmentation plus a 3-seed ensemble
  (`gru_soft_aug_ens`), aimed at the t+3 macro-F1 interval.

### 2026-10-29: Major update 6 (revised paper)

- `paper/revised_paper.md`: the full revised manuscript. Every number
  comes from `results/`, and a reproduction table maps each figure and
  table to its script.
- `make_figures.py` regenerates the 4 paper figures (300 dpi, IEEE
  column width).
- Corrected reference [2] (authors and venue, marked to verify). Ref. [7]
  still needs authors.

### 2026-10-29: Major update 5 (exact-label metrics ≥ 85%)

- Proved the ceiling for blind exact-beam prediction: the non-causal
  oracle reaches 79.0% / 0.59.
- New `beam_training_overhead.py`: two-stage reduced beam training with
  five shortlist methods, nested-CV operating points, and
  sequence-bootstrap CIs. At t+1, exact top-1 is 98.1–98.8% and macro-F1
  0.890–0.898, measuring 4–5.6 of 64 beams.
- `beam_sequence_model.py` now saves out-of-fold scores for downstream
  use.
