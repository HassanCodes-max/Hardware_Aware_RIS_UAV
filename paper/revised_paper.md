# Hardware-Aware AI-Assisted Beam Training, Tracking, and Channel Estimation for RIS-Assisted mmWave UAV Links

*Revised draft v2 (2026-10-29). Every number below is produced by the scripts in this repository; the
section "Reproducing this paper" at the end maps each table and figure to its script. Items marked
**[VERIFY]** need the author's confirmation before submission.*

**Maavia**, Department of Computer Science, Institute of Space Technology (IST), Islamabad, Pakistan

---

## Abstract

Millimetre-wave (mmWave) and sub-terahertz links can give unmanned aerial vehicles (UAVs) multi-gigabit
connectivity, but narrow beams, high path loss and UAV mobility make beam management costly, and
practical hardware departs from the ideal models most reconfigurable intelligent surface (RIS) studies
assume. We present a hardware-aware framework for an RIS-assisted 60 GHz UAV link that models
transmitter power-amplifier (PA) distortion through the Bussgang decomposition, RIS phase quantization,
phase noise, amplitude–phase coupling and antenna mismatch, together with ITU-R P.676-13 gaseous
absorption. On this model we show that (i) impairments cost 7.8 dB of SNDR, dominated by amplitude–phase
coupling, so a realistic RIS needs about 2.25× more elements than an ideal one for full coverage of a
300 m flight; (ii) a position-parametric channel estimator reaches within 0.15 dB of perfect channel
knowledge with 64× fewer pilots than per-element LMMSE, and needs no pilots when a 100 ms Kalman
prediction is available; and (iii) predictive tracking delivers 99.3% of the ideal array gain at 500 ms
latency, whereas persistence delivers 63%. On real 60 GHz measurements (DeepSense 6G Scenario 23) we
show that exact blind beam prediction is bounded at 79% top-1 by measurement noise, and propose an
AI-assisted reduced beam training scheme that selects the exact optimal beam 98.1% of the time
(macro-F1 0.890, 95% CI lower bound 0.853) while measuring 4 of 64 beams, under sequence-level nested
cross-validation.

**Index Terms**: mmWave, sub-THz, UAV, reconfigurable intelligent surface, beam training, beam
tracking, channel estimation, hardware impairments, deep learning.

---

## I. Introduction

Sixth-generation (6G) systems target mmWave and sub-THz bands for their large bandwidths [1], [2].
UAVs benefit directly, as aerial base stations, relays and sensing platforms, but high-frequency links
depend on narrow beams that a moving UAV quickly leaves. Exhaustive beam sweeping costs time that a
fast link cannot spare, which has motivated learning-based beam prediction from position and vision
[3], [4]. Reconfigurable intelligent surfaces (RISs) add a programmable reflection path when the direct
link is blocked [14], but real RIS hardware and transceivers are not ideal: transceiver distortion and
RIS phase noise change both performance and the optimal configuration [5], [6], and a practical RIS
element couples its reflection amplitude to its phase [15].

Three gaps motivate this work. First, RIS–UAV studies often pair idealised hardware with idealised
link budgets, which hides the hardware cost. Second, phase-tracking studies frequently score
controllers by per-element phase error, a metric that, as we show, says nothing about received power.
Third, beam prediction results on public datasets are easy to overstate: in the dataset we use, the
ground-truth beam is the argmax of the same power sweep often used as input, and consecutive samples
are nearly identical, so random splits leak.

**Contributions.**

1. A hardware-aware RIS–UAV link model with transmitter PA distortion (Bussgang SNDR), RIS phase
   quantization, phase noise, amplitude–phase coupling and S11 mismatch, on a full ITU-R P.676-13
   absorption model, with an impairment ablation and an RIS sizing rule (Sec. II, VI-A).
2. Hardware-aware control and estimation: a coupling-aware discrete phase search, an LMMSE estimator
   that inverts the hardware's realised training patterns, and a position-parametric estimator with an
   adaptive pilot budget (Sec. III, VI-B).
3. A predictive tracking study scored by delivered array gain, showing that position-based RIS control
   works at realistic sensor precision once the right metric is used (Sec. IV, VI-C).
4. On real 60 GHz measurements, a proof that exact blind beam prediction is noise-limited, a GRU trained
   on a soft power target, and an AI-assisted reduced beam training scheme meeting ≥ 85% exact top-1
   and macro-F1 at 6.3% of the sweep overhead, evaluated with sequence-level nested cross-validation
   and bootstrap confidence intervals (Sec. V, VI-D).

---

## II. System Model

### A. Geometry

A fixed base station (BS) at (25, −15, 10) m serves a UAV through an N-element RIS whose direct path is
blocked. The RIS is a 32×32 uniform planar array with λ/2 spacing (8 cm aperture at 60 GHz), mounted on
a facade at (0, 0, 20) m with outward normal +x; it serves only the half-space in front of it. The UAV
flies at 10 m/s; trajectories cover straight, diagonal, climbing, descending, circular, helical and
waypoint paths, all in front of the panel.

### B. Channel

For element n at distance d, the BS→RIS and RIS→UAV coefficients are

h₁[n] = √G_BS · (λ / 4πd₁ₙ) · e^(−j2πd₁ₙ/λ) · a(d₁ₙ) · √(G_e cos^q θ₁ₙ),   (1)

and likewise h₂[n] with the UAV gain G_UAV, where a(d) = 10^(−γd/20) is the gaseous-absorption amplitude
with γ from the ITU-R P.676-13 line-by-line model (44 O₂ and 35 H₂O lines) at 1013.25 hPa, 288.15 K and
7.5 g/m³ [8]; γ = 14.6 dB/km at 60 GHz. G_e = π is the gain of a (λ/2)² element [16], θ is measured
from the panel normal, q = 1.5, and the pattern is zero behind the panel. G_BS = 25 dBi and
G_UAV = 10 dBi; the transmit power is 20 dBm and the receiver noise is kTB·NF = −87 dBm (B = 100 MHz,
NF = 7 dB).

The received signal with reflection coefficients Γₙ is

y = Σₙ h₂[n] Γₙ h₁[n] · x + w.   (2)

With ideal hardware the optimal phases are φₙ = −∠(h₁[n] h₂[n]) and the array gain grows as N²
(verified: +12.04 dB per quadrupling of N).

### C. Hardware Impairments

**Transmitter PA.** A passive RIS has no amplifier; the PA sits at the BS. We apply Saleh's AM/AM and
AM/PM model [9] (α = 2.1587, β = 1.1517, α_φ = 4.0033, β_φ = 9.1040) to a complex Gaussian (OFDM-like)
input at 6 dB input back-off. By the Bussgang decomposition [17] the output is K·x + d, with d
uncorrelated with x, so a fraction ρ_d of the output power is distortion that travels through the
channel with the signal [18]:

SNDR = ρ_u P|h|² / (ρ_d P|h|² + σ²).   (3)

At 6 dB back-off ρ_u/ρ_d gives an SNDR ceiling of 13.3 dB, independent of link strength.

**RIS.** Commanded phases are quantized to b = 2 bits, perturbed by Gaussian phase noise (σ = 5°), and
realised with the phase-dependent amplitude of a practical element [15]:

β(φ) = (1 − β_min) · ((sin(φ − ϕ) + 1)/2)^κ + β_min,   β_min = 0.2, ϕ = 0.43π, κ = 1.6,   (4)

scaled by a reflection efficiency of 0.85.

**Antenna mismatch.** S11 = −10 dB at the BS and UAV ports gives a power transfer of 1 − |S11|² each.

Each impairment is switchable, so ideal and hardware-aware links are compared on identical geometry.

---

## III. Hardware-Aware Control and Channel Estimation

### A. Coupling-Aware Phase Selection

Because (4) ties amplitude to phase, the ideal phases are no longer optimal. For a common output phase ψ
the best realisable level for element n maximises Re(cₙ Γ(level) e^(−jψ)), with cₙ = h₁[n]h₂[n]; we
sweep 64 values of ψ and keep the best. The search is fully vectorised (≈3 ms for N = 1024).

### B. Per-Element Estimation: LS vs Hardware-Aware LMMSE

The RIS cycles through N Hadamard training patterns, each held for L pilot symbols. The *naive* LS
estimator assumes the commanded 0/π states were realised exactly. Under (4) the two states have
amplitudes 0.19 and 0.91, which biases LS. The *hardware-aware* estimator trains on the antipodal state
pair with the largest reflected energy, models the realised coefficients through (4), and applies
LMMSE with an i.i.d. prior; because PA distortion is signal-dependent, only its average power enters the
noise model.

### C. Position-Parametric Estimation

The BS and RIS are fixed, so c(p) = h₁ ⊙ h₂(p) depends only on the UAV position p. Beyond the Fresnel
distance (≈5 m here) the pilots constrain the direction of p sharply and its range weakly. Starting from
a prior p₀, the RIS transmits M random realisable patterns A, and the receiver estimates the direction
by maximising the likelihood under an unknown complex gain,

p̂ = argmax_p |yᴴ A c(p)|² / ‖A c(p)‖²,   (5)

over direction cosines on a coarse grid finer than the beam width, then a fine grid. The RIS is then
configured from c(p̂) with the search of Sec. III-A.

**Adaptive pilot budget.** The search succeeds when the matched-filter pilot SNR M·L·SNR_coh/N exceeds a
threshold; SNR_coh is predicted from p₀ and the known geometry. With M = 64 we choose L to reach a
threshold of 128, selected on random seeds separate from those reported. If the prior's angular
uncertainty is already below one eighth of the beam width, no pilots are sent.

---

## IV. Predictive RIS Tracking

The RIS must be configured for where the UAV will be once measurement, computation and control latency
have passed. We predict the position h steps ahead (h·10 ms, h ∈ {1, 10, 50}) from RTK-class position
measurements (σ = 2 cm per axis) with four predictors: persistence, constant-velocity extrapolation, a
constant-velocity Kalman filter [13] whose acceleration PSD is tuned on training trajectories, and a
ridge regressor on an 8-sample window trained on 60 random trajectories. All predictors see the same
noisy measurements, and the nine designed trajectories are held out.

**Metric.** Mean absolute per-element phase error counts a phase offset common to all elements as error,
but such an offset leaves |Σₙ cₙ e^(jφ̂ₙ)| unchanged. We therefore score the fraction of the ideal
coherent gain delivered at the true position:

G = |Σₙ cₙ e^(jφ̂ₙ)|² / (Σₙ |cₙ|)².   (6)

---

## V. AI Beam Prediction on Real Measurements

### A. Dataset and Evaluation Protocol

DeepSense 6G Scenario 23 [10] provides 11,387 samples of 64-beam received-power sweeps at 60 GHz from a
drone flight campaign, organised into 35 flight sequences, with the optimal-beam index as label (56
classes in use). Three properties of the data shape the protocol:

1. **The label is the argmax of the sweep** for all 11,387 samples, so predicting the current beam from
   the current sweep is trivial (100% by argmax). Every task below predicts the beam at time t + h from
   sweeps up to time t.
2. **Consecutive samples are near-duplicates** (≈7 Hz). All models are evaluated with GroupKFold(5)
   over flight sequences, so a test sequence is never seen in training.
3. **Near-ties are noise.** In 75.3% of sweeps the best and second-best beams are within 0.25 dB (they
   are adjacent in 90.6%), while the same beam's power changes by a median 0.18 dB between consecutive
   sweeps.

### B. Blind Prediction

A two-layer GRU (hidden size 128) reads the last 8 sweeps (dB relative to peak) and the sampling gaps,
and outputs beam logits added to a learned multiple of the last sweep, so that it learns a residual over
persistence. Because near-tied labels are noise, we train against a **soft target**,
softmax(P_{t+h}/0.5 dB), the measured power of the future sweep, rather than its argmax; a hard-label GRU
is the ablation. Early stopping uses held-out training sequences. Random-forest and extra-trees
baselines use the last 3 sweeps.

**Noise ceiling.** As an upper bound for any causal predictor of the exact label, a non-causal oracle
that averages the sweeps *before and after* the target sweep reaches only **79.0% exact top-1 and 0.59
macro-F1**.

### C. AI-Assisted Reduced Beam Training

A deployed link does not have to choose blind: it can measure a few candidate beams at time t + h and
keep the strongest. The predictor ranks the beams; the system measures the top K and keeps the best;
the result is scored against the **full 64-beam ground truth** on exact top-1 and macro-F1, with an
overhead of K/64 of a full sweep. Shortlists are produced from out-of-fold GRU scores and compared with
non-learned rankings (the last sweep's strongest beams; a window around the last best beam). The fused
ranking adds the z-scored GRU log-probabilities to the z-scored last sweep with equal weight. An
adaptive variant measures the smallest set whose GRU probability reaches τ.

**Operating points by nested cross-validation.** For each held-out fold, K (or τ) is the cheapest
setting meeting 85% on both metrics over the other four folds, and is then applied unchanged to the
held-out fold. 95% confidence intervals come from 500 bootstrap resamples of whole flight sequences.

---

## VI. Results

### A. Link Budget and Hardware Impairments

With ideal hardware the SNR is 15.8 dB at 20 m and 8.8 dB at 50 m. At 50 m the combined impairments cost
**7.83 dB** (Fig. 1): amplitude–phase coupling 5.51 dB, PA nonlinearity 1.46 dB, S11 mismatch 0.92 dB,
2-bit quantization 0.91 dB and phase noise 0.03 dB. Coupling-aware phase selection (Sec. III-A)
recovers about 0.5 dB with 2-bit phases.

![Fig. 1](figures/fig_impairment_ablation.png)

*Fig. 1. SNDR penalty of each impairment alone and combined, UAV at 50 m.*

Impairments change how large the RIS must be (Fig. 2). Over a 20 → 320 m flight, a 64×64 RIS gives 100%
availability (SNDR ≥ 0 dB) with ideal hardware but only 57% with realistic hardware; a 96×96 RIS
(24 cm) restores 100%. The hardware cost is therefore about 2.25× in element count.

![Fig. 2](figures/fig_ris_sizing.png)

*Fig. 2. Link availability over the flight versus RIS size.*

**Note on the first draft.** Without antenna and element gains and with an 8×8 (2 cm) RIS, the SNR is
−43.2 dB at 20 m, the value reported in our earlier draft; it reflects missing gains, not sub-THz
propagation. The same cause explains the earlier −170 dBm at 140 GHz: with gains included, an 8×8 RIS
receives −115.9 dBm at 140 GHz at 50 m, and the N² law holds at both carriers (Table I).

*Table I. Received power at 50 m, ideal hardware (dBm).*

| RIS | 60 GHz | 140 GHz |
|---|---|---|
| 8×8 | −102.3 | −115.9 |
| 16×16 | −90.2 | −103.8 |
| 32×32 | −78.2 | −91.8 |
| 64×64 | −66.2 | −79.7 |

### B. Channel Estimation

Naive LS saturates at about −7.5 dB NMSE because of the coupling bias, whereas hardware-aware LMMSE keeps
improving with pilot length (−11.1 dB at 20 m with 4096 symbols per pattern) and is never worse. In
closed loop (estimate, configure, measure SNDR), the hardware-aware pipeline beats the naive one at
every distance and pilot length, by 0.4–1.2 dB.

Per-element estimation, however, scales poorly with range. The parametric estimator (Table II) reaches
perfect-CSI SNDR to within 0.15 dB at 20–100 m with 64×1024 = 65k pilot symbols, against 4.2M for
per-element LMMSE, which still falls 1.6 dB short at 100 m. A 5 m GPS prior alone is useless
(−23 to −31 dB) because its error exceeds the beam width. With the adaptive budget (Table III), the
estimator delivers 96.5–100% of perfect-CSI SNR on average at every range and prior, and needs no pilots
when the 100 ms Kalman prediction is used as prior.

*Table II. SNDR (dB) from a 5 m GPS prior.*

| UAV at | Perfect CSI | GPS only | Parametric, 65k pilots | Per-element LMMSE, 4.2M pilots |
|---|---|---|---|---|
| 20 m | 7.47 | −23.7 | 7.47 | 7.32 |
| 50 m | 1.50 | −31.4 | 1.49 | 1.07 |
| 100 m | −4.75 | −22.9 | −4.90 | −6.33 |

*Table III. Adaptive parametric estimation: mean (worst of 10) % of perfect-CSI SNR, and pilot symbols.*

| UAV at | GPS prior (5 m) | Kalman, 500 ms (31 cm) | Kalman, 100 ms (5 cm) |
|---|---|---|---|
| 20 m | 97.4 (90.9), 6k | 97.3 (93.4), 4k | 99.5 (98.6), 0 |
| 50 m | 98.4 (95.9), 31k | 98.5 (95.8), 33k | 99.9 (99.4), 0 |
| 100 m | 96.9 (93.5), 131k | 96.8 (91.7), 131k | 99.9 (99.6), 0 |
| 200 m | 96.5 (86.3), 524k | 99.4 (98.5), 0 | 100.0 (99.8), 0 |

### C. Predictive Tracking

Every predictor shows a mean absolute phase error of 89–92°, the value an unaligned phase profile would
give, yet the delivered gain (6) ranges from 63% to 100% (Table IV, Fig. 3). The phase-error metric
therefore cannot rank controllers. Prediction matters at realistic latencies: at 500 ms the Kalman
filter delivers 99.3% of the ideal gain (5th percentile 97.4%), the learned ridge predictor 98.8%,
constant velocity 71.4% and persistence 63.0%. All predictors run in under 0.1 ms per step.

*Table IV. Delivered array gain (%) and position RMSE (cm), held-out trajectories.*

| Horizon | Persistence | Constant velocity | Kalman CV | Ridge (learned) |
|---|---|---|---|---|
| 10 ms | 99.9 / 10.4 | 99.9 / 7.7 | **100.0 / 2.0** | 100.0 / 2.7 |
| 100 ms | 94.0 / 97.9 | 97.2 / 51.9 | **99.98 / 5.2** | 99.9 / 8.2 |
| 500 ms | 63.0 / 488 | 71.4 / 250 | **99.3 / 30.7** | 98.8 / 41.4 |

![Fig. 3](figures/fig_tracking_gain.png)

*Fig. 3. Delivered array gain versus prediction horizon.*

### D. Beam Prediction on Scenario 23

**Blind prediction** (Table V). At t + 1 (≈140 ms) the soft-target GRU selects a beam within the 0.25 dB
noise margin 95.6% of the time (class-balanced 89.2%), has 95.1% top-3 accuracy, and delivers 98.9% of
the optimal received power. Its exact top-1 is 70.9% against the 79.0% oracle ceiling. Trained on hard
labels, the same GRU falls below persistence at t + 3; the soft target turns it into the best model at
both horizons.

*Table V. Blind beam prediction, GroupKFold over 35 sequences.*

| Horizon | Model | Within noise | Balanced within noise | Top-3 | Power ratio | Exact top-1 | Macro-F1 |
|---|---|---|---|---|---|---|---|
| t+1 | Persistence | 93.7 | 86.3 | 94.5 | 98.6 | 69.2 | 0.45 |
| t+1 | RF + sweep prior | 93.7 | 86.7 | 94.9 | 98.6 | 70.4 | 0.41 |
| t+1 | GRU, hard label | 93.7 | 82.3 | 94.6 | 98.6 | 70.9 | 0.41 |
| t+1 | **GRU, soft target** | **95.6** | **89.2** | **95.1** | **98.9** | **70.9** | **0.45** |
| t+3 | Persistence | 78.1 | 66.1 | 80.3 | 93.7 | 53.3 | 0.26 |
| t+3 | GRU, hard label | 73.2 | 57.9 | 80.6 | 93.1 | 49.5 | 0.21 |
| t+3 | **GRU, soft target** | **81.1** | **65.8** | **84.8** | **95.5** | **55.3** | **0.24** |

**Reduced beam training** (Table VI, Fig. 4). Measuring a handful of predicted beams resolves the
near-ties that blind prediction cannot. At t + 1, the fused GRU + last-sweep shortlist selects the exact
optimal beam 98.1% of the time with macro-F1 0.890 over all 56 classes, measuring 4 of 64 beams (6.3% of
a full sweep); both 95% intervals lie above 0.85. The adaptive shortlist reaches 98.8% / 0.898 with 5.6
beams on average.

*Table VI. Reduced beam training, nested-CV operating points, scored against the full-sweep ground
truth [95% CI over sequences].*

| Horizon | Shortlist | Beams measured | Exact top-1 | Macro-F1 (56 classes) |
|---|---|---|---|---|
| t+1 | Last-sweep ranking | 4.2 | 97.6 [96.4, 98.5] | 0.871 [0.829, 0.912] |
| t+1 | Window around last beam | 6.0 | 99.0 [98.4, 99.6] | 0.929 [0.890, 0.980] |
| t+1 | **GRU + last sweep** | **4.0** | **98.1 [97.2, 98.7]** | **0.890 [0.853, 0.929]** |
| t+1 | **GRU adaptive (τ = 0.8)** | **5.6** | **98.8 [98.4, 99.3]** | **0.898 [0.857, 0.953]** |
| t+3 | Last-sweep ranking | 16.0 | 97.9 [96.6, 98.8] | 0.860 [0.823, 0.906] |
| t+3 | GRU, soft target | 10.8 | 97.1 [95.2, 98.7] | 0.861 [0.804, 0.928] |

![Fig. 4](figures/fig_beam_training.png)

*Fig. 4. Exact top-1 (left) and macro-F1 (right) at t + 1 versus beams measured, out-of-fold
predictions; dotted line: non-causal blind-prediction ceiling.*

The learned ranking's advantage is in measurement cost rather than in the attainable accuracy: a
non-learned window of 6 beams also exceeds 85% at t + 1, while the AI shortlists reach the target with
4–5.6 beams, and at t + 3 the GRU needs 10.8 beams against 16 for the last-sweep ranking. At t + 3 the
macro-F1 point estimates exceed 0.85 but their intervals do not; classes with at most 5 samples drive
the gap (macro-F1 over classes with ≥ 5 samples is 0.94–0.96).

---

## VII. Discussion

**Metrics shape conclusions.** Two conclusions of our earlier draft reversed under correct metrics. A
Random Forest reported at 80.9% on the current-beam task was below the 100% that argmax achieves, and
"open-loop phase tracking is impossible at sub-THz" followed from per-element phase error, which is
blind to the common phase offset. We recommend that RIS tracking studies report delivered gain and that
beam-prediction studies on argmax-labelled data predict future beams under sequence-level splits.

**Where hardware awareness pays.** The largest impairment is amplitude–phase coupling. With 2-bit phases
coupling-aware control recovers only 0.5 dB, because the four available states leave little room; the
estimation side gains more (LMMSE removes the coupling bias), and the largest system-level effect is
the 2.25× element count needed for coverage.

**Learning where it helps.** On power sweeps alone the GRU's gain over persistence is modest at t + 1 and
clearer at t + 3. The soft target matters more than the architecture. Logit adjustment for the class
imbalance [19] lowered every metric because the beam distribution differs between flights.

**Limitations.** Saleh and coupling coefficients are literature values, not measurements of a 60 GHz
PA or RIS. The obliquity exponent is assumed. The physical-layer results are simulated; the
beam-prediction results use power sweeps only, because the UAV GPS files of Scenario 23 were not
available to us at the time of writing. The t + 3 macro-F1 is close to 0.85 and its interval includes
it. The 140 GHz branch is used only for link budget and scaling.

---

## VIII. Conclusion

We built a hardware-aware model of an RIS-assisted mmWave UAV link and used it, with real 60 GHz
measurements, to evaluate beam management end to end. Hardware impairments cost 7.8 dB and 2.25× RIS
elements; a position-parametric estimator matches perfect channel knowledge with 64× fewer pilots;
Kalman-based prediction preserves 99.3% of the array gain at 500 ms latency; and AI-assisted reduced beam
training selects the exact optimal beam 98.1% of the time at 6.3% of the sweep cost, with all reported
metrics evaluated under sequence-level cross-validation. Future work will add UAV position to beam
prediction, measured hardware coefficients, and joint training of the tracker and the beam shortlist.

---

## References

[1] S. M. Shah *et al.*, "TeraHertz communication and associated challenges in 6G cellular networks,"
in *Proc. ICCIS*, 2021, doi: 10.1109/ICCIS54243.2021.9676382.

[2] C. Chaccour, M. N. Soorki, W. Saad, M. Bennis, P. Popovski, and M. Debbah, "Seven defining features
of terahertz (THz) wireless systems: A fellowship of communication and sensing," *IEEE Commun. Surveys
Tuts.*, vol. 24, no. 2, pp. 967–993, 2022. **[VERIFY: the first draft attributed this paper to
I. F. Akyildiz et al. in IEEE JSAC.]**

[3] G. Charan *et al.*, "Towards real-world 6G drone communication: Position and camera aided beam
prediction," in *Proc. IEEE GLOBECOM*, 2022, pp. 2951–2956.

[4] V. A. Nugroho and B. M. Lee, "GPS-aided deep learning for beam prediction and tracking in UAV mmWave
communication," *IEEE Access*, vol. 13, pp. 117065–117077, 2025.

[5] J. Dai, F. Zhu, C. Pan, H. Ren, and K. Wang, "Statistical CSI-based transmission design for
reconfigurable intelligent surface-aided massive MIMO systems with hardware impairments," *IEEE Wireless
Commun. Lett.*, vol. 11, no. 1, pp. 38–42, Jan. 2022.

[6] M. Meles, A. Rajasekaran, L. Mela, and R. Jäntti, "RIS-assisted three-dimensional drone localization
and tracking under hardware impairments," *IEEE Access*, vol. 12, pp. 81348–81361, 2024.

[7] **[VERIFY: authors missing]** "Fundamentals of wobbling and hardware impairments-aware air-to-ground
channel model," *IEEE Trans. Veh. Technol.*, vol. 73, no. 12, pp. 17946–17962, Dec. 2024.

[8] Recommendation ITU-R P.676-13, "Attenuation by atmospheric gases and related effects," ITU, 2022.

[9] A. A. M. Saleh, "Frequency-independent and frequency-dependent nonlinear models of TWT amplifiers,"
*IEEE Trans. Commun.*, vol. COM-29, no. 11, pp. 1715–1720, Nov. 1981.

[10] A. Alkhateeb *et al.*, "DeepSense 6G: A large-scale real-world multi-modal sensing and
communication dataset," *IEEE Commun. Mag.*, vol. 61, no. 9, pp. 122–128, Sep. 2023.

[11] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[12] K. Cho *et al.*, "Learning phrase representations using RNN encoder-decoder for statistical machine
translation," in *Proc. EMNLP*, 2014.

[13] R. E. Kalman, "A new approach to linear filtering and prediction problems," *J. Basic Eng.*,
vol. 82, no. 1, pp. 35–45, 1960.

[14] Q. Wu and R. Zhang, "Towards smart and reconfigurable environment: Intelligent reflecting surface
aided wireless network," *IEEE Commun. Mag.*, vol. 58, no. 1, pp. 106–112, Jan. 2020.

[15] S. Abeywickrama, R. Zhang, Q. Wu, and C. Yuen, "Intelligent reflecting surface: Practical phase
shift model and beamforming optimization," *IEEE Trans. Commun.*, vol. 68, no. 9, pp. 5849–5863,
Sep. 2020.

[16] W. Tang *et al.*, "Wireless communications with reconfigurable intelligent surface: Path loss
modeling and experimental measurement," *IEEE Trans. Wireless Commun.*, vol. 20, no. 1, pp. 421–439,
Jan. 2021.

[17] J. J. Bussgang, "Crosscorrelation functions of amplitude-distorted Gaussian signals," MIT Research
Laboratory of Electronics, Tech. Rep. 216, 1952.

[18] E. Björnson, J. Hoydis, and L. Sanguinetti, "Massive MIMO networks: Spectral, energy, and hardware
efficiency," *Found. Trends Signal Process.*, vol. 11, no. 3–4, pp. 154–655, 2017.

[19] A. K. Menon *et al.*, "Long-tail learning via logit adjustment," in *Proc. ICLR*, 2021.

---

## Reproducing this paper

| Item | Script | Output |
|---|---|---|
| Sec. VI-A, Fig. 1–2, Table I | `run_physical_layer.py` | `results/physical_layer_metrics.json` |
| Sec. VI-B, Tables II–III | `run_physical_layer.py` (`parametric_study`, `adaptive_study`, `closed_loop_study`) | same |
| Sec. VI-C, Table IV, Fig. 3 | `tracking_eval.py` | `results/tracking_metrics.json` |
| Sec. VI-D, Table V | `beam_prediction.py`, `beam_sequence_model.py` | `results/beam_*_metrics.csv` |
| Sec. VI-D, Table VI, Fig. 4 | `beam_training_overhead.py` (after `beam_sequence_model.py`) | `results/beam_training_overhead.json` |
| All figures | `make_figures.py` | `paper/figures/` |
