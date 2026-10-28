# Audit of the first draft (code, dataset, paper)

Date: 2026-10-29. Scope: every `.py` file, the Scenario 23 data in
`public_data/`, and `Research_Paper_IEEE_final.pdf` (first draft).

Each finding has evidence you can rerun and a status. Severity:

- **Critical**: invalidates a result or conclusion in the paper.
- **High**: wrong physics or a result that can't be reproduced.
- **Medium**: correctness or hygiene issue that doesn't change conclusions.

---

## Critical

### C1. The beam label leaks from the input features

`beam_label == argmax(power_1..power_64) + 1` for **all 11,387 samples**.
So "predict the optimal beam from the current 64-beam power sweep" (paper
Sec. III-B, Table II) needs no model: `argmax` scores **100%**. The Random
Forest's 80.90% is *worse* than that one-liner.

- Evidence: `tests/test_audit_facts.py::test_scenario23_label_is_argmax_of_power`,
  and `legacy_checks` in `results/beam_prediction_metrics.json`.
- Fix: `beam_prediction.py` now evaluates the task a tracker actually faces.
  Given the sweeps up to time t, it picks the beam for t+1 or t+3 *before*
  that sweep happens.

### C2. Temporal leakage from a random train/test split

Samples come at about 7 Hz, so consecutive samples are nearly identical.
A random 80/20 split puts each test sample's neighbours into training.

- Fix: every model is evaluated with `GroupKFold(5)` over the dataset's
  own `seq_index`, so the 35 flight sequences never span train and test.

### C3. The tracking conclusion rests on a metric that can't measure it

Paper Sec. IV / Table IV / Discussion: "sub-THz RIS phase control cannot
rely on open-loop position tracking at any achievable sensor precision".
That conclusion comes from **mean absolute per-element phase error**, which
counts a *common* phase offset on all elements as error. A common offset
does not change the received power at all.

The correct metric is the fraction of the ideal coherent array gain that
the predicted phases deliver at the true position:
`|Σ g_n e^{jφ̂_n}|² / (Σ|g_n|)²`.

| Predictor (2 cm sensor noise) | Abs. phase error | Array gain delivered |
|---|---|---|
| Persistence, 10 ms ahead | 88.5° ("chance") | **99.76%** |
| Kalman CV, 10 ms ahead | 90.4° ("chance") | **99.91%** |
| Kalman CV, 500 ms ahead | 92.0° | **98.31%** |
| Persistence, 500 ms ahead | 90.1° | 58.0% |

Every method shows about 90° "phase error" but delivers almost the full
gain. Open-loop position-based control works. Prediction matters at
realistic control latencies (≥100 ms), where Kalman beats persistence
by 40 points. Source: `tracking_eval.py` → `results/tracking_metrics.json`.

### C4. The paper describes code that isn't in the repository

There is no code for any of these: the Saleh PA, S11 mismatch, obliquity
factor, Kalman filter, both GRUs, the RGB CNN, the multimodal model, the
140 GHz branch, and the element-scaling experiment. Tables II (rows 2–5),
III and IV, and the ≈29 dB / 2.53 dB numbers in Sec. VI-B can't be
reproduced from this repo.

- Fix: the PA, S11, obliquity, Kalman, scaling and 140 GHz pieces are now
  implemented (`hardware.py`, `channel.py`, `tracking_eval.py`,
  `run_physical_layer.py`). The GRU, CNN and multimodal models are not
  rebuilt yet: PyTorch isn't installed, and the images aren't available
  (see H3).

### C5. The −43 dB SNR comes from missing antenna gains, not from sub-THz physics

The v1 link has **0 dBi antennas at both the BS and the UAV**, isotropic RIS
elements, and an 8×8 RIS at λ/2, which is only **2 cm × 2 cm**. Real 60 GHz
links use 20–30 dBi arrays. The paper's sentence "quantitatively
reproducing the severe sub-THz attenuation" misattributes the cause.

The "unexpectedly low −170 dBm" at 140 GHz (Sec. VI-D), which the paper
blames on a normalisation problem, has the same cause. With gains
included, 8×8 at 140 GHz gives −115.9 dBm.

- Fix: `config.py` adds BS/UAV antenna gains (25/10 dBi), the RIS element
  gain (π, i.e. 4πA/λ²), a 32×32 default RIS, and noise from kTB·NF.
  `SimulationConfig.paper_v1()` reproduces the old numbers exactly
  (−43.23 dB) so every change can be traced.

### C6. The geometry puts the UAV behind or edge-on to the RIS

In v1 the RIS lies flat at z = 20 m with its normal along z. The BS sits
directly below it at the origin, and the UAV flies at z = 20 m, which is
in the panel's plane. Most designed trajectories fly *above* the panel,
which is behind a reflector lit from below. The isotropic model let the
RIS "reflect" into its back half-space.

The paper's "newly quantified grazing-angle effect (≈29 dB)" is an
artefact of this choice, not a general finding.

- Fix: the RIS is now facade-mounted (normal +x). The cos^q pattern applies
  on both hops, and the back half-space gets zero signal (tested).

---

## High

### H1. PA nonlinearity was put on the RIS

A passive RIS has no power amplifier. The PA belongs at the BS
transmitter. A fixed "2.53 dB" loss also can't capture what PA
nonlinearity actually does: the distortion travels with the signal and
caps the SNDR however strong the link is.

- Fix: `hardware.py` applies Saleh's model (both AM/AM and AM/PM; the paper
  only gave α, β for AM/AM) at the BS with input back-off. It uses the
  Bussgang decomposition to give useful and distortion power. At 6 dB IBO
  the SNDR ceiling is 13.3 dB.

### H2. The tracking baselines were not comparable

- The paper's Kalman RMSE of 12.1 cm with 2 cm measurement noise is *worse
  than persistence* (10 cm). A correctly tuned constant-velocity KF gives
  2.0 cm, so the original filter was almost certainly buggy. The paper
  also calls it an "EKF", but the model is linear.
- Constant-velocity extrapolation at 0.1 cm only makes sense on
  noise-free positions. On the same 2 cm noisy measurements it gives
  7.7 cm. So CV probably saw ground truth while the KF saw noise. This
  can't be confirmed because the code isn't in the repo.
- Fix: in `tracking_eval.py` all predictors see the same noisy
  measurements. The Kalman q is tuned on training trajectories only, and
  the learned predictor trains on random trajectories and is tested on
  the designed ones.

### H3. The dataset in the repo is incomplete

- Only **699 of 11,387** camera images are present. The `mmWave_data/` power
  files and all `unit2/` files (UAV GPS, altitude, speed, pitch, roll) are
  missing.
- So `build_scenario23_dataset.py` can't rebuild the beam CSV, the RGB/CNN
  results can't be reproduced, and **UAV GPS is never used**, even though
  position-aided beam prediction is the standard approach for this scenario.
- Action needed: download the full Scenario 23 development set (see
  README → Next steps).

### H4. The sequence count doesn't match the dataset

The paper builds "28 sequences" using a >2 s gap rule. The dataset ships an
official `seq_index` with **35** sequences, monotonic in row order. We now
use `seq_index`.

### H5. Phase-wrap statement

"Phase wraps every λ/2 ≈ 2.5 mm at 60 GHz": λ at 60 GHz is 5.0 mm, and
the phase wraps once per λ of *path-length* change. This is minor, and
C3 makes the argument moot anyway.

---

## Medium

| # | Issue | Status |
|---|---|---|
| M1 | 60 GHz is mmWave. "Sub-THz" usually means 100–300 GHz, but the title and abstract claim sub-THz while the main results are at 60 GHz. | Paper wording: open |
| M2 | A 286 MB `scenario23_beam_model.pkl` was committed through LFS, and it can be regenerated. | Untracked, added to `.gitignore` (local file kept) |
| M3 | `requirements.txt` listed only numpy. | Fixed |
| M4 | README was outdated (it said "circular trajectory" and was missing half the files). | Rewritten |
| M5 | `phase_tracking.py` crashed if `results/` didn't exist. | Fixed |
| M6 | `ris_complex_channel` left out atmospheric loss but `physical_ris_received_power_w` included it. | Unified |
| M7 | Noise was hard-coded to 1e-12 W. | Now kTB·NF (−87 dBm at 100 MHz, NF 7 dB) |
| M8 | The ITU-R P.676 sum ran 128× per time step in Python loops. | Cached once per config; channel vectorised |
| M9 | The abstract says BER and channel-estimation error are evaluated, but they were future work. | Implemented: BER (QPSK/16-QAM vs SNDR), LS/LMMSE NMSE |
| M10 | Ref. [7] has no authors; ref. [2] has no volume or pages. | Paper: open |
| M11 | Legacy beam scripts duplicated the same split and model code. | Moved to `legacy/` |

### Verified correct (no change needed)

- `atmosphere.py` (ITU-R P.676-13 line-by-line) gives 14.6 dB/km at 60 GHz,
  35.6 dB/km at 183 GHz and 0.69 dB/km at 140 GHz, which matches the
  Recommendation's curves.
- The coherent-combining phase rule (Eq. 3) and the N² array scaling are
  correct (`test_coherent_gain_scales_with_n_squared`).

---

## About the "85% minimum on all metrics" target

Every metric that measures what the system delivers is now at or above
85% at the operating point. Two classification metrics can't reach 85%
honestly on this data. Forcing them up would mean leaking labels again.

**At or above 85%** (beam numbers at t+1 with the GRU soft target;
tracking with Kalman; estimation adaptive):

| Metric | Value |
|---|---|
| Beam within measurement noise (0.25 dB) of best | 95.6% |
| Beam within noise, class-balanced | 89.3% |
| Beam top-3 accuracy | 95.1% |
| Beam ±1 accuracy | 91.5% |
| Beam received-power ratio | 98.9% |
| Tracking: array gain at 10 / 100 / 500 ms | 100.0 / 99.98 / 99.3% |
| Channel estimation: % of perfect-CSI SNR (mean, every range and prior) | 96.5–100% |
| Link availability, 96×96 RIS, realistic hardware | 100% |

**Below 85%, and why:**

| Metric | Value | Why |
|---|---|---|
| Exact top-1 beam index (t+1) | 70.9% | In 75% of sweeps the best two beams are within 0.25 dB, while the same beam fluctuates by a median 0.18 dB between sweeps. The exact index is decided by noise. |
| Macro-F1 on exact index (t+1) | 0.45 | Same label noise, plus rare beams that appear in only a few of the 35 sequences. The class-balanced within-noise accuracy (89.3%) measures the same thing without the noise. |
| t+3 (~0.4 s ahead) beam metrics | 66–85% | Power ratio is 95.5%. The other beam metrics need UAV GPS, which is blocked on the dataset download. |

For the paper, report t+1 as the operating point and the noise-aware
metrics as primary. Put the exact top-1 and macro-F1 next to them with
this explanation. A reviewer who runs `argmax` on the data will check
exactly this.
