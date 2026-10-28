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
| `tracking_eval.py` | Position predictors (persistence, CV, Kalman, learned) scored by delivered array gain at 10/100/500 ms horizons | `tracking_metrics.{json,csv}` | ~1 min |
| `main.py` | Per-time-step trace along the default flight | `simulation.csv` | ~4 s |
| `phase_tracking.py`, `dataset_generator.py`, `hybrid_dataset_generator.py` | Synthetic phase datasets | `*.csv` | minutes |

`results/` is generated and is not tracked by git.

## Project structure

| File | Role |
|---|---|
| `config.py` | All parameters. `SimulationConfig()` is the corrected v2 model; `SimulationConfig.paper_v1()` reproduces the first draft; `.ideal()` turns off all impairments. |
| `channel.py` | Channels H1/H2, coherent phases, `link_metrics` (SNR, SNDR, SE, BER), `beamforming_gain_ratio`, `estimate_cascaded_channel` (naive LS vs hardware-aware LMMSE) |
| `hardware.py` | Saleh PA + Bussgang factors, S11, RIS quantization, phase noise, amplitude–phase coupling, `hardware_aware_phases` |
| `ris.py` | RIS element positions for any panel orientation |
| `atmosphere.py`, `p676_data/` | ITU-R P.676-13 Annex 1 line-by-line absorption |
| `trajectory.py` | Straight, circular, helical, waypoint and random UAV trajectories |
| `build_scenario23_dataset.py`, `public_data_adapter.py` | Scenario 23 loading (rebuilding needs the full download, see Next steps) |
| `public_data/scenario23_dev/` | Scenario 23 manifest + extracted 64-beam power CSV (only 699 of 11,387 images present) |
| `legacy/` | First-draft Random Forest scripts, kept so Table II can be reproduced |
| `tests/` | Regression tests for the physics and for the audit facts |

## Current results (2026-10-29)

**Beam prediction** (Scenario 23, GroupKFold over 35 sequences, power sweeps only, no GPS yet)

| Horizon | Model | Top-1 | Top-3 | ±1 beam | Power ratio | Power loss |
|---|---|---|---|---|---|---|
| t+1 (~140 ms) | Persistence | 69.2% | 94.5% | 91.4% | 98.6% | 0.06 dB |
| | RF + sweep prior | 70.4% | 94.9% | 89.6% | 98.6% | 0.06 dB |
| | **GRU, soft target** | **70.9%** | **95.1%** | **91.5%** | **98.9%** | **0.05 dB** |
| t+3 | Persistence | 53.3% | 80.3% | 76.7% | 93.7% | 0.28 dB |
| | RF + sweep prior | 53.9% | 84.2% | 75.3% | 94.4% | 0.25 dB |
| | GRU, hard label | 49.5% | 80.6% | 70.2% | 93.1% | 0.31 dB |
| | **GRU, soft target** | **55.3%** | **84.8%** | **78.1%** | **95.5%** | **0.20 dB** |

The GRU is trained against softmax(measured dB / 0.5 dB) of the future
sweep instead of the argmax label, because near-tied beams make the hard
label mostly noise. With hard labels the GRU loses to persistence at t+3,
as the first draft's GRU did. With the soft target it is the best model.
The GRU uses an 8-sweep window and the trees a 3-sweep window, so their
sample sets differ slightly at sequence edges. Top-1 stays capped by
near-tied beams (see AUDIT.md).

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
4. [ ] **Rewrite the paper** per AUDIT.md:
   - Table II: new task, split and metrics.
   - Sec. IV/VI-E: the array-gain metric; the conclusion is reversed.
   - Sec. VI-B: new link budget and the impairment ablation.
   - Sec. VI-D: the 140 GHz numbers are resolved.
   - Fix the "sub-THz" wording at 60 GHz and references [2] and [7].
5. [x] Closed loop: estimate, then control, then measure SNDR
   (`closed_loop_study`). Next, a parametric (angle-based) estimator,
   which should close most of the gap to perfect CSI at 100 m.
6. [ ] **Hardware parameters**: replace the textbook Saleh and coupling
   coefficients with values from a D-band PA / 60 GHz RIS datasheet or
   paper, and run the sweep at 140 GHz.
7. [x] Designed trajectories moved in front of the RIS.
8. [ ] Higher-resolution RIS phases (3-bit) and a coupling-aware
   continuous optimiser. This is where hardware-aware control should gain
   more than 0.5 dB.
