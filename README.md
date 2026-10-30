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
python -m pytest tests          # 14 regression tests, a few seconds
```

## Experiments

| Script | What it does | Output (in `results/`) | Runtime |
|---|---|---|---|
| `run_physical_layer.py` | Link budget, ideal vs naive vs hardware-aware, impairment ablation, RIS scaling (60/140 GHz), RIS sizing, channel-estimation NMSE | `physical_layer_metrics.json` | ~25 s |
| `beam_prediction.py` | Next-beam prediction on Scenario 23, cross-validated over flight sequences | `beam_prediction_metrics.{json,csv}`, `beam_prediction_folds.csv` | ~3 min |
| `beam_sequence_model.py` | GRU over 8 sweeps (hard label vs soft power target), same protocol and metrics | `beam_sequence_metrics.{json,csv}` | ~15 min (CPU) |
| `beam_training_overhead.py` | Two-stage AI-assisted beam training: shortlist K beams from the past, measure them, keep the best. Scored against the full 64-beam ground truth with nested-CV operating points and sequence-bootstrap CIs (run after `beam_sequence_model.py`) | `beam_training_overhead.json`, `*_curves.csv` | ~3 min |
| `conformal_beam_training.py` | Conformal beam shortlists calibrated on held-out flights to a target exact top-1 (90–99%); rank and APS scores (run after `beam_sequence_model.py`) | `conformal_beam_training.{json,csv}` | ~1 min |
| `closed_loop_learned_tracking.py` | Ten deep models trained on measured-only logs, run in the loop, against the tie graph at equal budget; CPU inference latency fp32/int8 | `closed_loop_learned_tracking.{json,csv}` | ~40 min (CPU) |
| `closed_loop_beam_tracking.py` | Replays every flight with policies that know only the beams they measured (window + learned tie partners, periodic or loss-triggered full sweeps); exact top-1, latency, effective spectral efficiency | `closed_loop_beam_tracking.{json,csv}`, `*_flights.csv` | ~1 min |
| `camera_beam_prediction.py` | Encodes every BS camera frame with a frozen ImageNet ResNet-18 (`--extract`, cached locally), then predicts the beam from the camera alone. `SCENARIO23_FEATURES=camera` / `gps+camera` feeds a 16-d PCA of it to the two prediction scripts | `camera_beam_prediction.json` | ~25 min extract (CPU), ~2 min study |
| `closed_loop_side_tracking.py` | Closed loop with camera (UAV pixel) and GPS candidates per step and loss-triggered sweeps; per-flight tests against the tie-graph tracker | `closed_loop_side_tracking.{json,csv}`, `*_flights.csv` | ~10 min |
| `significance_tests.py` | Paired per-flight tests (flight bootstrap CI, Wilcoxon, Holm) for closed-loop policies and for each side-information run | `significance_tests.{json,csv}` | ~10 s |
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
| `make_figures.py`, `paper/` | Regenerates every paper figure from `results/`; `paper/revised_paper.md` is the revised manuscript, `paper/build_docx.py` turns it into `paper/revised_paper_ieee.docx` (IEEE layout) |
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
| | GRU, soft target | 95.6% | 89.3% | 95.1% | 91.5% | 98.9% | *70.9%* | *0.45* |
| | **GRU ensemble, augmented** | **96.4%** | **91.7%** | **96.1%** | **93.4%** | **99.1%** | *72.1%* | *0.48* |
| t+3 (~0.4 s) | Persistence | 78.1% | 66.1% | 80.3% | 76.7% | 93.7% | *53.3%* | *0.26* |
| | RF + sweep prior | 78.5% | 63.0% | 84.2% | 75.3% | 94.4% | *53.9%* | *0.21* |
| | GRU, soft target | 81.1% | 65.8% | 84.8% | 78.1% | 95.5% | *55.3%* | *0.24* |
| | **GRU ensemble, augmented** | **86.0%** | **75.5%** | **86.9%** | **83.8%** | **96.8%** | *56.7%* | *0.28* |

The GRU is trained against softmax(measured dB / 0.5 dB) of the future
sweep instead of the argmax label. With hard labels it loses to
persistence at t+3, as the first draft's GRU did. The GRU uses an
8-sweep window and the trees a 3-sweep window, so their sample sets
differ slightly at sequence edges.

The italic columns score the exact beam index. That index is decided by
noise in 75% of sweeps (see AUDIT.md), so these columns can't reach 85%
on this data and aren't used as targets. They are reported so reviewers
can see them.

**AI-assisted reduced beam training** (the deployed task; exact labels; target ≥ 95%)

Blind prediction of the exact beam index has a hard ceiling. Even a
non-causal oracle that averages the sweeps *before and after* the target
reaches only **79.0% exact top-1 and 0.59 macro-F1**. So the system
predicts a shortlist, measures only those beams at the target time, and
keeps the strongest. It is scored against the full 64-beam ground truth.
Every setting is chosen by nested CV: for each held-out fold, the
lowest-latency setting meeting 95% on exact top-1 and all-class macro-F1
over the other four folds. The tie graph (below) is always fitted on folds
other than the ones scored, including inside that selection. The 95% CIs
come from bootstrapping over the 35 flight sequences.

**Learned tie graph.** This codebook's strong beams aren't contiguous:
beams 27, 28, 32, 33 and 38 are often within 0.1 dB of each other, so
"look around the last beam" misses the true best. For each beam, the
training flights give the beams most often within 1 dB of it when it was
best. *Two-round* measures a shortlist, then the graph partners of the
measured winner (plus a full-sweep fallback if power drops > 3 dB).
*Burst* adds the partners of the two *predicted* best beams to the
shortlist, so everything goes out in one burst with no report round trip.

**Latency model** (5G NR FR2, 120 kHz SCS): 64 SSB beams per 5 ms
half-frame, so 78 µs per beam, plus 1 ms per extra report round trip. A
full sweep costs 5 ms. GRU inference (0.5 ms single, 1.4 ms 3-model
ensemble, batch 1, CPU) is **off the critical path**: the shortlist for
t+h is computed from sweep t during the ~140 ms gap between sweeps.

| Horizon | Method | Beams (of 64) | Latency, mean | Exact top-1 [95% CI] | Macro-F1, all 56 classes [95% CI] | Macro-F1, classes with ≥ 5 samples |
|---|---|---|---|---|---|---|
| t+1 | Last sweep, K=4 (old 85% operating point) | 4.2 | 0.33 ms | 97.6% [96.4, 98.5] | 0.871 [0.829, 0.912] | 0.945 |
| t+1 | Last sweep, K=6 (no AI) | 6.0 | 0.47 ms | **99.1%** [98.6, 99.5] | **0.960** [0.940, 0.985] | 0.981 |
| t+1 | **GRU ensemble + tie graph, two-round** | 7.0 | 0.55 ms | **99.3%** [98.7, 99.7] | **0.961** [0.941, 0.990] | 0.985 |
| t+1 | GRU ensemble + sweep, burst | 6.4 | 0.50 ms | **99.2%** [98.7, 99.6] | **0.955** [0.940, 0.977] | 0.984 |
| t+3 | Last sweep, K=16 (no AI) | 16.0 | 1.25 ms | 97.9% [96.6, 98.8] | 0.860 [0.823, 0.906] | 0.956 |
| t+3 | GRU ensemble, single round | 15.2 | 1.19 ms | 98.9% [98.0, 99.5] | 0.896 [0.864, 0.944] | 0.965 |
| t+3 | Last sweep + tie graph, two-round | 13.9 | 2.11 ms | **99.4%** [98.9, 99.7] | 0.914 [0.889, 0.956] | **0.979** |
| t+3 | **GRU ensemble + tie graph, two-round** | 11.7 | 1.93 ms | **99.1%** [98.3, 99.7] | 0.929 [0.905, 0.964] | **0.974** |
| t+3 | GRU ensemble, burst | 15.6 | 1.22 ms | **98.9%** [98.1, 99.6] | 0.912 [0.880, 0.949] | **0.971** |

All rows also reach ≥ 99.5% within-noise accuracy and ≥ 99.8% of the
best beam's power.

- **t+1 meets 95% on every metric**, CIs included for exact top-1, at
  0.5 ms of beam training (10× faster than a full sweep). Raising the
  target from 85% to 95% costs 2–3 more beams. At this horizon the
  non-AI K=6 shortlist also meets the target, so the honest AI gain at
  t+1 is small.
- **t+3 meets 95% on exact top-1 (99%) and on macro-F1 over classes with
  ≥ 5 samples (0.97–0.98)**, but not on all-class macro-F1 (0.91–0.93,
  CIs reaching 0.95–0.96). The gap is three beams seen **once** in the
  whole dataset (23, 34, 55) whose true best beats the runner-up by
  ~0.01 dB. Each costs 1/56 ≈ 0.018 of macro-F1. Pooled curves reach
  0.95 at 16–18 beams, but nested selection on four folds under-provisions
  because one singleton flips the selection-fold score. We report the
  nested numbers.
- The tie graph is the main gain at t+3: all-class macro-F1 rises from
  0.86–0.90 to 0.91–0.93 with fewer beams. The GRU ensemble + tie graph
  is the cheapest t+3 setting (11.7 beams).

**Position- and vision-aided prediction** (full Scenario 23 download; paper
Tables XI-XII, Fig. 7). Side information: UAV GPS and kinematics; a frozen
ResNet-18 embedding of the BS camera frame; or the UAV's pixel position,
found without labels (frame minus blurred sky, darkest 3x3 spot; 54 ms per
frame). Blind prediction, within noise / balanced within noise, %:

| Horizon | Model | Power only | + GPS | + camera emb. | + UAV pixel | + GPS + UAV pixel |
|---|---|---|---|---|---|---|
| t+1 | GRU ensemble | 96.4 / 91.7 | 96.5 / 91.6 | 95.9 / 89.6 | 96.5 / 91.4 | **96.6 / 91.4** |
| t+3 | RF + sweep prior | 78.5 / 63.0 | **82.7 / 65.3** | 75.9 / 61.0 | 78.1 / 62.8 | 81.4 / 65.3 |
| t+3 | GRU, soft target | 81.1 / 65.8 | **83.2 / 69.7** | 74.8 / 59.0 | 79.0 / 64.1 | 82.8 / 67.9 |
| t+3 | GRU ensemble | 86.0 / 75.5 | 87.2 / 79.3 | 80.5 / 67.3 | 84.5 / 73.9 | **87.3 / 77.4** |

No side information significantly improves blind prediction per flight
(GPS at t+3: +1.1 points, Wilcoxon p = 0.34, Holm 0.68). The camera
embedding hurts (-5.6 points at t+3, worse on 23 of 33 flights). Alone, the
UAV pixel finds the current beam within noise 81% of the time (embedding
67%, most frequent beam 15%). With GPS, reduced beam training at t+3
reaches 99.3% exact top-1 and all-class macro-F1 0.946 (power only 0.929)
with 14.5 instead of 11.7 beams; CIs overlap. The other tables stay
power-only (reproducible from the shipped CSV): `SCENARIO23_FEATURES=power`
ignores `unit2/`. The `unit2_height` channel (31-391) doesn't match the
<= 120 m BS distance, so it is used but not interpreted.

**Sensing-aided closed loop** (`closed_loop_side_tracking.py`, paper
Table XIV, Fig. 9). Each step adds the top-k beams of a random forest fed
the previous step's UAV pixel and/or GPS; full sweeps only on beam loss.

| Policy (33 flights) | Beams | Latency | Exact top-1 | SE % of genie |
|---|---|---|---|---|
| Full sweep every step | 64.0 | 5.00 ms | 100% | 96.5 |
| Tracker, 4 tie partners, full every 8 | 13.2 | 1.05 ms | 97.2% | 99.1 |
| Camera top-4 + tracker, loss-triggered | 7.4 | 0.59 ms | 98.6% | 99.5 |
| GPS top-4 + tracker, loss-triggered | 7.3 | 0.59 ms | 99.1% | 99.6 |
| **Camera + GPS top-4 + tracker, loss-triggered** | **7.0** | **0.56 ms** | **99.2%** | **99.6** |

Camera + GPS beats the tracker on 25 flights and loses on 1 (+2.0 points,
95% CI +1.3 to +2.9), with lower latency on 30 of 33.
Closed-loop macro-F1 (all classes / classes with >= 5 steps): tracker
0.843 / 0.934, camera + GPS top-4 0.944 / 0.980, so the sensing-aided
tracker meets 95% on exact top-1 and macro-F1 (>= 5) at 0.56 ms.

Locator check against GPS (`--validate`, no labels): pixel -> east/north
on held-out flights gives R^2 0.67 / 0.12, median 4.4 m (height is not
recorded consistently, so a pixel fixes direction, not distance); about
1% of frames are misdetected (pixel jumps > 100 px in 1.8% of frame pairs).

**140 GHz RIS size vs receiver noise figure** (`run_physical_layer.py`,
paper Table III): full availability needs 24 cm at NF 6 dB, 28 cm at
7 dB, 32 cm at 10 dB (the default, conservative; measured D-band SiGe
LNAs alone: 4.8-6.1 dB).

**Significance** (`significance_tests.py`, paper Table XIII). Over 33
flights: the 4-partner tracker loses 2.8 points of exact top-1 vs a full
sweep every step but gains 2.6 points of spectral efficiency on 32 of 33
flights; 6 partners beat 4 by 2.0 points (26 better, 0 worse); 8 add
nothing significant over 6.

**Conformal shortlists** (open-loop, like the table above). A measured
set returns the exact best beam if and only if that beam is in the set,
so set coverage *is* exact top-1. Calibrating the set size on four folds
and testing on the fifth:

| Target exact top-1 | t+1: achieved (worst fold), beams | t+3: achieved (worst fold), beams |
|---|---|---|
| 90% | 91.5% (89.6%), 2.2 | 92.4% (89.4%), 4.0 |
| 95% | **96.1% (95.3%), 3.0** | **95.5% (92.7%), 5.4** |
| 98% | 98.3% (97.2%), 4.4 | 98.0% (96.8%), 8.8 |
| 99% | 99.2% (98.7%), 6.2 | 99.0% (98.1%), 17.4 |

Coverage tracks the target on average. Flights differ, so a single held-out
fold can fall up to 2.3 points short at t+3. Adaptive (APS) sets over-cover
(98.8% for a 95% target) with a 95th-percentile size of 41–56 beams, so
they are worse for worst-case latency.

**Closed loop** (`closed_loop_beam_tracking.py`; see AUDIT.md H6). The
tables above build each shortlist from full sweeps that a reduced-training
system would not have. Here every policy knows only the beams it measured.
SE is effective spectral efficiency (training airtime removed, beam ageing
to the next step measured), as % of a genie with the best beam and no
training time. The dataset's powers are unitless, so SE is given at
nominal best-beam SNRs.

| Policy | Exact top-1 | Within noise | Beam-training latency | SE @ 0 / 10 / 20 dB |
|---|---|---|---|---|
| Full sweep every step | 100% | 100% | 5.00 ms | 96.5 / 96.5 / 96.5% |
| Window ±1, full sweep every 4 | 91.2% | 96.4% | 1.50 ms | 98.2 / 98.5 / 98.7% |
| **Window ±1 + 4 tie partners, full every 4** | **97.8%** | **99.4%** | 1.61 ms | 98.7 / 98.8 / 98.8% |
| **Window ±1 + 4 tie partners, full every 8** | **97.2%** | **99.2%** | **1.05 ms** | **99.0 / 99.1 / 99.2%** |
| Window ±1, loss-triggered sweeps only | 75.5% | 83.2% | 0.35 ms | 92.9 / 95.0 / 96.8% |
| Window ±1 + 6 tie partners, loss-triggered only | 96.1% | 96.9% | 0.64 ms | 98.5 / 98.9 / 99.2% |

- The learned tie graph is what makes closed-loop tracking work: with
  loss-triggered sweeps only it lifts exact top-1 from 75.5% to 96.1%.
- At this testbed's 142 ms step a full sweep is only 3.5% of airtime, so
  throughput gains are a few percent. The gain is in latency per
  decision: about 1 ms instead of 5 ms.
- **Learned models in the loop** (`closed_loop_learned_tracking.py`, paper
  Table X / Fig. 6): ten models (GRU, LSTM, TCN, Transformer, MLP; GRU
  hidden size, learning rate, window 4/8/16, soft/hard target) trained on
  measured-only logs add k model beams to the tracker. At equal budget the
  tie graph wins: 99.16% / 99.28% exact top-1 with +2 / +4 tie partners
  against 98.17–98.71% / 98.59–98.83% for the models. The
  validation-selected Transformer is no better on test. MLP inference
  0.11 ms (fp32, one laptop CPU thread); int8 was slower for every model.

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

## Next steps (what we have to do)

Ordered by impact. Check items off here as they're done.

1. [x] **Download the full Scenario 23 dev set (manual: needs sign-in).** Done 2026-10-30;
   kept local (git-ignored). Results: paper Table XI / Fig. 7, `results/gps/`.
   Create a free account at deepsense6g.net → Scenarios → Scenario 23 →
   download the dev set. Unzip it so `unit1/` and `unit2/` sit inside
   `public_data/scenario23_dev/`, merging with what is already there.
   Then run `python beam_prediction.py` and `python beam_sequence_model.py`;
   both pick up the GPS features automatically.
2. [x] Position + power features wired in (`load_scenario23_kinematics`).
   Waiting on step 1. Goal: top-3 ≥ 90% at t+3.
3. [x] Sequence model: GRU with soft target (`beam_sequence_model.py`).
   Next, try a TCN or Transformer once GPS is available.
4. [x] Paper rewritten: `paper/revised_paper.md`, and the IEEE Word
   version `paper/revised_paper_ieee.docx` (built by
   `paper/build_docx.py`). All [VERIFY] items resolved (2026-10-30):
   [20] full author list and pages, [21] replaced by a measured D-band
   SiGe LNA (NF 4.8-6.1 dB; the model's 10 dB is now called a conservative
   receiver NF, with a 6/7/10 dB sensitivity in Table III), [22] pinned to
   TS 38.213 V17.9.0.
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
6. [x] D-band PA and LNA values taken from published results; coupling
   swept because no measured 60 GHz curve was found. Still open: a
   measured 60 GHz RIS unit-cell response, if the lab can provide one.
7. [x] Designed trajectories moved in front of the RIS.
8. [x] Phase-resolution sweep done (up to 1.01 dB gain at 4 bits).
