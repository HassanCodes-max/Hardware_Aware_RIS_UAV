# Hardware-Aware RIS UAV

Hardware-aware, AI-driven beam tracking and channel estimation for a
RIS-assisted 60 GHz (with 140 GHz exploration) UAV link.

The project has two halves:

1. **Physical layer**: BS → RIS → UAV channel with ITU-R P.676-13 gaseous
   absorption, element patterns, and hardware impairments (Saleh PA at the
   BS, RIS phase quantization, phase noise, amplitude–phase coupling, S11
   mismatch). It compares ideal hardware, a naive controller and a
   hardware-aware controller and estimator.
2. **AI tracking**: proactive beam prediction on real measurements
   (DeepSense 6G Scenario 23), and predictive RIS phase tracking from noisy
   UAV positions.

Read [AUDIT.md](AUDIT.md) first. It explains what was wrong in the first
draft and why several experiments were redesigned.

## Setup

```powershell
cd Hardware_Aware_RIS_UAV
python -m pip install -r requirements.txt
python -m pytest tests          # 7 regression tests, < 1 s
```

## Experiments

| Script | What it does | Output (in `results/`) | Runtime |
|---|---|---|---|
| `run_physical_layer.py` | Link budget, ideal vs naive vs hardware-aware, impairment ablation, RIS scaling (60/140 GHz), RIS sizing, channel-estimation NMSE | `physical_layer_metrics.json` | ~25 s |
| `beam_prediction.py` | Next-beam prediction on Scenario 23, cross-validated over flight sequences | `beam_prediction_metrics.{json,csv}`, `beam_prediction_folds.csv` | ~3 min |
| `beam_sequence_model.py` | GRU over 8 sweeps (hard label vs soft power target), same protocol and metrics | `beam_sequence_metrics.{json,csv}` | ~15 min (CPU) |
| `beam_training_overhead.py` | Two-stage AI-assisted beam training: shortlist K beams from the past, measure them, keep the best. Scored against the full 64-beam ground truth with nested-CV operating points and sequence-bootstrap CIs (run after `beam_sequence_model.py`) | `beam_training_overhead.json`, `*_curves.csv` | ~3 min |
| `tracking_eval.py` | Position predictors (persistence, CV, Kalman, learned) scored by delivered array gain at 10/100/500 ms horizons | `tracking_metrics.{json,csv}` | ~1 min |
| `main.py` | Per-time-step trace along the default flight | `simulation.csv` | ~4 s |
| `phase_tracking.py`, `dataset_generator.py`, `hybrid_dataset_generator.py` | Synthetic phase datasets | `*.csv` | minutes |

`results/` is generated and is not tracked by git.

## Project structure

| File | Role |
|---|---|
| `config.py` | All parameters. `SimulationConfig()` is the corrected v2 model; `SimulationConfig.paper_v1()` reproduces the first draft; `.ideal()` turns off all impairments. |
| `parametric_estimation.py` | Position-parametric estimator: GPS prior + M pilot patterns → refine the UAV direction → phases from g(p̂) |
| `channel.py` | Channels H1/H2, coherent phases, `link_metrics` (SNR, SNDR, SE, BER), `beamforming_gain_ratio`, `estimate_cascaded_channel` (naive LS vs hardware-aware LMMSE) |
| `hardware.py` | Saleh PA + Bussgang factors, S11, RIS quantization, phase noise, amplitude–phase coupling, `hardware_aware_phases` |
| `ris.py` | RIS element positions for any panel orientation |
| `atmosphere.py`, `p676_data/` | ITU-R P.676-13 Annex 1 line-by-line absorption |
| `trajectory.py` | Straight, circular, helical, waypoint and random UAV trajectories |
| `build_scenario23_dataset.py`, `public_data_adapter.py` | Scenario 23 loading (rebuilding needs the full download, see Next steps) |
| `public_data/scenario23_dev/` | Scenario 23 manifest + extracted 64-beam power CSV (only 699 of 11,387 images present) |
| `legacy/` | First-draft Random Forest scripts, kept so Table II can be reproduced |
| `make_figures.py`, `paper/` | Regenerates every paper figure from `results/`; `paper/revised_paper.md` is the revised manuscript |
| `tests/` | Regression tests for the physics and for the audit facts |

## Current results (2026-10-29)

**Beam prediction** (Scenario 23, GroupKFold over 35 sequences, power sweeps only, no GPS yet)

"Within noise" means the chosen beam is within 0.25 dB of the best beam.
That is the testbed's own noise floor: the same beam's power changes by a
median 0.18 dB between consecutive sweeps, so closer beams can't be told
apart. "Balanced" averages that accuracy per beam, so rare beams count
as much as common ones (the role macro-F1 plays for exact labels).

| Horizon | Model | Within noise | Balanced within noise | Top-3 | ±1 beam | Power ratio | *Exact top-1* | *Macro-F1* |
|---|---|---|---|---|---|---|---|---|
| t+1 (~140 ms) | Persistence | 93.6% | 86.3% | 94.5% | 91.4% | 98.6% | *69.2%* | *0.45* |
| | RF + sweep prior | 93.7% | 86.7% | 94.9% | 89.6% | 98.6% | *70.4%* | *0.41* |
| | **GRU, soft target** | **95.6%** | **89.2%** | **95.1%** | **91.5%** | **98.9%** | *70.9%* | *0.45* |
| t+3 (~0.4 s) | Persistence | 78.1% | 66.1% | 80.3% | 76.7% | 93.7% | *53.3%* | *0.26* |
| | RF + sweep prior | 78.5% | 63.0% | 84.2% | 75.3% | 94.4% | *53.9%* | *0.21* |
| | **GRU, soft target** | **81.1%** | **65.8%** | **84.8%** | **78.1%** | **95.5%** | *55.3%* | *0.24* |

The GRU is trained against softmax(measured dB / 0.5 dB) of the future
sweep instead of the argmax label. With hard labels it loses to
persistence at t+3, as the first draft's GRU did. The GRU uses an
8-sweep window and the trees a 3-sweep window, so their sample sets
differ slightly at sequence edges.

The italic columns score the exact beam index. That index is decided by
noise in 75% of sweeps (see AUDIT.md), so these columns can't reach 85%
on this data and aren't used as targets. They are reported so reviewers
can see them.

**AI-assisted reduced beam training** (the deployed task; exact labels)

Blind prediction of the exact beam index has a hard ceiling. Even a
non-causal oracle that averages the sweeps *before and after* the target
reaches only **79.0% exact top-1 and 0.59 macro-F1**. So the system
predicts a shortlist, measures only those beams at the target time, and
keeps the strongest. It is scored against the full 64-beam ground truth.
K (or the confidence threshold) for each held-out fold is chosen on the
other four folds. The 95% CIs come from bootstrapping over the 35 flight
sequences.

| Horizon | Shortlist | Beams measured (of 64) | Exact top-1 [95% CI] | Macro-F1, all 56 classes [95% CI] |
|---|---|---|---|---|
| t+1 | Last sweep (no AI) | 4.2 (6.6%) | 97.6% [96.4, 98.5] | 0.871 [0.829, 0.912] |
| t+1 | Window around last beam (no AI) | 6.0 (9.4%) | 99.0% [98.4, 99.6] | 0.929 [0.890, 0.980] |
| t+1 | **GRU + last-sweep fusion** | **4.0 (6.3%)** | **98.1% [97.2, 98.7]** | **0.890 [0.853, 0.929]** |
| t+1 | **GRU adaptive (stop at 80% confidence)** | 5.6 (8.7%) | **98.8% [98.4, 99.3]** | **0.898 [0.857, 0.953]** |
| t+3 | Last sweep (no AI) | 16.0 (25%) | 97.9% | 0.860 [0.823, 0.906] |
| t+3 | GRU soft | 10.8 (16.9%) | 97.1% | 0.861 [0.804, 0.928] |

At t+1, the operating point, both AI shortlists meet 85% on exact top-1
and all-class macro-F1 with the whole CI above 0.85, while measuring
16× fewer beams than a full sweep. The plain window is also strong on
macro-F1. The AI's measurable gain is fewer measurements: 4 beams vs 6
at t+1, and 10.8 vs 16 (a third less) at t+3. At t+3 the macro-F1 point
estimates pass but the CIs straddle 0.85. Classes with ≤ 5 samples
dominate that gap; macro-F1 over classes with ≥ 5 samples is 0.94–0.96.
Logit adjustment for class imbalance was tried and rejected because it
lowered every metric (see the script docstring).

**Predictive RIS tracking** (32×32 RIS, 2 cm position noise, held-out trajectory types)

| Horizon | Persistence | Constant velocity | Kalman CV | Learned (Ridge) |
|---|---|---|---|---|
| 10 ms | 99.9% | 99.9% | **100.0%** | 100.0% |
| 100 ms | 94.0% | 97.2% | **99.98%** | 99.9% |
| 500 ms | 63.0% | 71.4% | **99.3%** | 98.8% |

The values are the mean fraction of the ideal array gain delivered, with
all 9 trajectories fully in front of the RIS.

**Closed loop: estimate, then control** (SNDR in dB, hardware-aware vs naive pipeline)

| UAV at | Pilots per pattern | Perfect CSI | Naive (LS + −∠ĝ) | Hardware-aware (LMMSE + aware phases) |
|---|---|---|---|---|
| 20 m | 1024 | 7.47 | 6.77 | **7.19** |
| 50 m | 4096 | 1.50 | 0.57 | **1.07** |
| 100 m | 4096 | −4.75 | −7.27 | **−6.33** |

The hardware-aware pipeline wins at every distance and pilot length, by
0.4–1.2 dB. Per-element estimation gets expensive with range, while
position-based tracking keeps more than 99% of the gain with no pilots.
This argues for setting phases from tracking and using pilots only to
refine them.

**Adaptive parametric estimation**: the pilot budget and search width are
set automatically from the prior. Values are the mean (worst of 10 trials)
% of perfect-CSI SNR, with pilot symbols used.

| UAV at | GPS prior (5 m) | Kalman prior, 500 ms (31 cm) | Kalman prior, 100 ms (5 cm) |
|---|---|---|---|
| 20 m | 97.4% (90.9%), 6k | 97.3% (93.4%), 4k | 99.5% (98.6%), 0 |
| 50 m | 98.4% (95.9%), 31k | 98.5% (95.8%), 33k | 99.9% (99.4%), 0 |
| 100 m | 96.9% (93.5%), 131k | 96.8% (91.7%), 131k | 99.9% (99.6%), 0 |
| 200 m | 96.5% (86.3%), 524k | 99.4% (98.5%), 0 | 100.0% (99.8%), 0 |

Per-element LMMSE uses 4.2M pilot symbols for a worse result.

**Parametric estimation, fixed budgets** (GPS prior with 5 m error per axis; SNDR in dB)

| UAV at | Perfect CSI | GPS only | Parametric, 16×256 = 4k pilots | Parametric, 64×1024 = 65k pilots | Per-element LMMSE, 1024×4096 = 4.2M pilots |
|---|---|---|---|---|---|
| 20 m | 7.47 | −23.7 | 7.32 | **7.47** | 7.32 |
| 50 m | 1.50 | −31.4 | −23.8 | **1.49** | 1.07 |
| 100 m | −4.75 | −22.9 | −47.9 | **−4.90** | −6.33 |

The cascaded channel depends only on the UAV's direction from the RIS
(3 unknowns, not 1024). Refining that direction from a few random pilot
patterns reaches near-perfect CSI with **64× fewer pilots** than
per-element LMMSE. Below a pilot-energy threshold the search locks onto
noise, as at 100 m with 4k pilots, so the budget must scale with range.
GPS alone can't steer a 32×32 RIS: 5 m of error is wider than the beam.

**Physical layer** (60 GHz, 32×32 RIS, 20 dBm, 25/10 dBi, UAV at 50 m)

- Ideal SNR is 8.8 dB. With all impairments the SNDR is 1.0 dB, a **7.8 dB
  hardware penalty**. The breakdown: amplitude–phase coupling 5.5 dB, PA
  (6 dB IBO) 1.5 dB, S11 0.9 dB, 2-bit quantization 0.9 dB, phase noise
  0.03 dB.
- Hardware-aware phase selection recovers about 0.5 dB with 2-bit phases.
- Availability (SNDR ≥ 0 dB) over the 20→320 m flight:

  | RIS | Ideal | Realistic hardware |
  |---|---|---|
  | 64×64 | 100% | 57% |
  | 96×96 | 100% | 100% |

  So impairments cost about 2.25× more RIS elements.
- Channel estimation (4096 pilots, 20 m): naive LS reaches −7.4 dB NMSE
  and is bias-limited. Hardware-aware LMMSE reaches **−11.1 dB**.

## Update log

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

## Next steps (what we have to do)

Ordered by impact. Check items off here as they're done.

1. [ ] **Download the full Scenario 23 dev set (manual: needs sign-in).**
   Create a free account at deepsense6g.net → Scenarios → Scenario 23 →
   download the dev set. Unzip it so `unit1/` and `unit2/` sit inside
   `public_data/scenario23_dev/`, merging with what is already there.
   Then run `python beam_prediction.py` and `python beam_sequence_model.py`;
   both pick up the GPS features automatically.
2. [x] Position + power features wired in (`load_scenario23_kinematics`).
   Waiting on step 1. Goal: top-3 ≥ 90% at t+3.
3. [x] Sequence model: GRU with soft target (`beam_sequence_model.py`).
   Next, try a TCN or Transformer once GPS is available.
4. [x] Paper rewritten: `paper/revised_paper.md`. Remaining: move it
   into the IEEE Word/LaTeX template, and resolve the [VERIFY] items.
   The original plan was:
   - Table II: new task, split and metrics.
   - Sec. IV/VI-E: the array-gain metric; the conclusion is reversed.
   - Sec. VI-B: new link budget and the impairment ablation.
   - Sec. VI-D: the 140 GHz numbers are resolved.
   - Fix the "sub-THz" wording at 60 GHz and references [2] and [7].
5. [x] Closed loop: estimate, then control, then measure SNDR
   (`closed_loop_study`).
5b. [x] Parametric (direction-based) estimator: within 0.15 dB of perfect
   CSI using 64× fewer pilots (`parametric_study`).
5c. [x] Adaptive pilot budget from the prior and predicted SNR, with
   GPS or Kalman priors (`adaptive_study`): ≥ 96.5% of perfect-CSI SNR
   on average, and 0 pilots with a 100 ms tracker prior.
6. [ ] **Hardware parameters**: replace the textbook Saleh and coupling
   coefficients with values from a D-band PA / 60 GHz RIS datasheet or
   paper, and run the sweep at 140 GHz.
7. [x] Designed trajectories moved in front of the RIS.
8. [ ] Higher-resolution RIS phases (3-bit) and a coupling-aware
   continuous optimiser. This is where hardware-aware control should gain
   more than 0.5 dB.
