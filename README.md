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

**Beam prediction** (Scenario 23, GroupKFold over 35 sequences, Random Forest + sweep prior)

| Horizon | Top-1 | Top-3 | ±1 beam | Power ratio | Power loss |
|---|---|---|---|---|---|
| t+1 (~140 ms) | 70.4% | 94.9% | 89.6% | 98.6% | 0.06 dB |
| t+3 | 53.9% | 84.2% | 75.3% | 94.4% | 0.25 dB |

Persistence ("keep the last beam") scores 69.0% / 94.4% / 98.6% at t+1, so
the learned models add only a little so far. Top-1 is capped by
near-tied beams (see AUDIT.md).

**Predictive RIS tracking** (32×32 RIS, 2 cm position noise, held-out trajectory types)

| Horizon | Persistence | Constant velocity | Kalman CV | Learned (Ridge) |
|---|---|---|---|---|
| 10 ms | 99.8% | 99.8% | **99.9%** | 99.9% |
| 100 ms | 91.3% | 94.3% | **99.8%** | 99.8% |
| 500 ms | 58.0% | 64.0% | **98.3%** | 96.9% |

The values are the mean fraction of the ideal array gain delivered.

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

## Next steps (what we have to do)

Ordered by impact. Check items off here as they're done.

1. [ ] **Download the full Scenario 23 dev set** from deepsense6g.net (all
   images, `mmWave_data/`, `unit2/` GPS/altitude/speed). This unblocks
   position-aided and image-aided beam prediction and rebuilding the CSV
   with `build_scenario23_dataset.py`.
2. [ ] **Position + power beam prediction**: add UAV GPS, altitude and
   velocity features to `beam_prediction.py`. Goal: beat persistence
   clearly at t+3 and beyond (top-3 ≥ 90%).
3. [ ] **Sequence model** (needs `pip install torch`): GRU/TCN over sweeps
   + GPS, with the same GroupKFold protocol. Report it against RF and
   persistence, not instead of them.
4. [ ] **Rewrite the paper** per AUDIT.md:
   - Table II: new task, split and metrics.
   - Sec. IV/VI-E: the array-gain metric; the conclusion is reversed.
   - Sec. VI-B: new link budget and the impairment ablation.
   - Sec. VI-D: the 140 GHz numbers are resolved.
   - Fix the "sub-THz" wording at 60 GHz and references [2] and [7].
5. [ ] **Closed loop**: estimate the channel with hardware-aware LMMSE,
   set phases from the estimate, and score the resulting SNDR. This
   connects estimation to tracking, as the paper title promises.
6. [ ] **Hardware parameters**: replace the textbook Saleh and coupling
   coefficients with values from a D-band PA / 60 GHz RIS datasheet or
   paper, and run the sweep at 140 GHz.
7. [ ] **Designed trajectories**: several still fly behind the
   facade-mounted RIS (`straight_y` sits exactly in its plane). Redesign
   them for the new geometry.
8. [ ] Higher-resolution RIS phases (3-bit) and a coupling-aware
   continuous optimiser. This is where hardware-aware control should gain
   more than 0.5 dB.
