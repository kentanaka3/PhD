# Dual-Track Epistemic Specification Matrix and Attention-Aligned Syntactic Framing

This reference establishes the **Dual-Track Epistemic Framework** for scientific manuscripts, separating deductive mathematical and theoretical physics reasoning from empirical, data-driven pipeline exposition.

---

## 1. Dual-Track Architecture Overview

Autoregressive language models generate tokens sequentially. Placing evidentiary qualifiers, governing principles, or telemetry markers at the **head of a clause** conditions attention heads before substantive assertions are emitted, eliminating speculative drift and epistemic conflation.

Scientific communication is divided into two parallel, non-overlapping tracks:
- **Track D (Deductive Track)**: Axiomatic mathematics, continuum physics, governing field equations, derivations, and formal proofs.
- **Track E (Empirical Track)**: Sensor telemetry, curated benchmark datasets, deterministic algorithms, machine-learning inferences, and qualitative engineering trade-offs.

---

## 2. Track D: Deductive Mathematics & Theoretical Physics

In theoretical physics and mathematics, truth is derived from axioms, conservation principles, and valid deductive logic—not from sensor calibration or statistical sample sizes.

| Tier | Category | Evidentiary Substrate | Mathematical Modifiers | Canonical Syntactic Anchor |
| :---: | :--- | :--- | :--- | :--- |
| **D1** | **Axioms & Definitions** | Mathematical definitions, algebraic structures, state spaces. | Domain $\Omega$, function space ($H^1, L^2$), state vector dimension, metric signature. | *"Let $(\Omega, \mathcal{F}, P)$ be a complete probability space, and define the state vector $\mathbf{x} \in \mathbb{R}^d$..."* |
| **D2** | **Governing Laws & PDEs** | Conservation laws (mass, momentum, energy), variational principles. | Stress tensor $\bm{\sigma}$, boundary conditions, constitutive parameters, source terms. | *"From conservation of linear momentum, the Cauchy stress tensor $\bm{\sigma}$ satisfies \cref{eq:momentum}..."* |
| **D3** | **Derivations & Proofs** | Step-by-step algebraic transformations, integration by parts, inequalities. | Governing equation labels, transformation lemma, machine precision (if numerical). | *"Substituting constitutive relation~\eqref{eq:hooke} into \cref{eq:momentum} and taking the divergence..."* |
| **D4** | **Asymptotic Regimes** | Asymptotic expansions, perturbation limits, scaling bounds. | Asymptotic condition ($N \to \infty$, $\omega \to \infty$, $Re \ll 1$), Landau symbols ($\mathcal{O}, o$). | *"In the high-frequency eikonal limit ($\omega \to \infty$), the phase function $\theta(\mathbf{x})$ satisfies..."* |

### Track D Syntactic Templates

#### Template D1: Functional Domain Specification
```latex
Let $\Omega \subset \mathbb{R}^3$ be an open bounded domain with Lipschitz boundary $\partial\Omega$.
We define the admissible displacement space as $\mathcal{V} = \{ \mathbf{u} \in [H^1(\Omega)]^3 : \mathbf{u}|_{\Gamma_D} = \mathbf{0} \}$.
```

#### Template D2: Governing Physical Conservation
```latex
Applying the principle of virtual work to the continuum body under traction $\mathbf{t}$
and body force $\mathbf{f}$, the weak formulation requires finding $\mathbf{u} \in \mathcal{V}$ such that
\cref{eq:weak_form} holds for all test functions $\mathbf{v} \in \mathcal{V}$.
```

#### Template D3: Step-by-Step Derivation Anchor
```latex
Substituting the constitutive stress-strain relation~\eqref{eq:hooke} into the momentum
balance equation~\eqref{eq:navier} and collecting terms of order $\mathcal{O}(\epsilon)$ yields...
```

---

## 3. Track E: Empirical Science & Data-Driven Pipeline

In experimental geophysics, data engineering, and machine learning, assertions must report physical telemetry, curated consensus metrics, or statistical uncertainty bounds.

| Tier | Category | Evidentiary Substrate | Mandatory Telemetry / Modifiers | Canonical Syntactic Anchor |
| :---: | :--- | :--- | :--- | :--- |
| **E1** | **Empirical Observations** | Raw sensor feeds, digitizer streams, lab measurements. | Instrument model, sampling rate ($f_s$), channel orientation, noise floor. | *"Continuous telemetry recorded by [Instrument] on channel [Ch] at \qty{[fs]}{\hertz}..."* |
| **E2** | **Ground Truth** | Curated reference benchmarks, expert labels. | Number of annotators ($K$), agreement metric (Fleiss' $\kappa$), arbitration rule. | *"Ground-truth arrivals were established through consensus review by $K=[N]$ analysts ($\kappa=[val]$)..."* |
| **E3** | **Deterministic Software** | Exact numerical algorithms, DSP filters, linear transforms. | Module source path, equation label, test fixture, numerical tolerance ($\epsilon$). | *"Evaluating the closed-form discrete transform in \cref{eq:[label]} via \texttt{[module]} ($\epsilon \le 10^{-12}$)..."* |
| **E4** | **Model Estimates** | Statistical fits, neural predictions, MCMC posteriors. | Sample size ($N$), point estimate, $95\%$ CI, standard error ($\mathrm{SE}$), seed hash. | *"Evaluating held-out test windows ($N=[val]$) with [Model] (checkpoint \texttt{[hash]}), the inferred parameter is..."* |
| **E5** | **Literature Baselines** | Published theoretical bounds, comparative models. | Citation key, bounding theorem, declared benchmark conditions. | *"In comparison with the established baseline of \cite{[key]}, which posits an optimal bound of..."* |
| **E6** | **Qualitative Analysis** | Engineering trade-offs, architecture rationale, threats to validity. | Governing operational assumptions, failure modes, boundary limitations. | *"We interpret this discrepancy as reflecting an operational trade-off between [A] and [B], assuming that..."* |

### Track E Syntactic Templates

#### Template E1: Sensor Telemetry Acquisition
```latex
Continuous three-component velocity telemetry was acquired by a broadband seismometer (passband \qtyrange{0.02}{50}{\hertz}) on channel \texttt{HHZ} at a sampling rate of \qty{100}{\hertz} with an effective dynamic range of \qty{140}{\decibel} (\cref{fig:station_tri}).
```

#### Template E2: Consensus Ground Truth
```latex
Reference P-wave arrival timestamps were curated by $K=3$ independent analysts following the ISC curation protocol, achieving inter-annotator agreement of Fleiss' $\kappa = 0.93$ with pairwise residuals bounded by $|t_i - t_j| \le \qty{40}{\milli\second}$.
```

#### Template E3: Deterministic Software Transformation
```latex
The raw waveforms were deterministically bandpass-filtered over the band \qtyrange{1.0}{20.0}{\hertz} using a 4th-order causal Butterworth filter implemented in \texttt{OGS/dsp/filter.py} (numerical stability verified against fixture \texttt{test_dsp} with $\epsilon \le 10^{-14}$).
```

#### Template E4: Statistical Model Prediction
```latex
Evaluating $N = 2500$ held-out test windows with the deep neural picker (checkpoint \texttt{v2.4}, seed $42$), the model inferred a mean arrival onset of $\hat{t} = \qty{14.23}{\second}$ with a $95\%$ confidence interval of $[\qty{14.18}{\second}, \qty{14.28}{\second}]$ (standard error $\mathrm{SE} = \qty{0.025}{\second}$).
```

---

## 4. Cross-Track Synthesis Paradigm

When transitioning from physical theory to empirical implementation within a single chapter or section, bridge Track D and Track E explicitly:

```latex
% [Track D: Theoretical Governing Law]
From elastodynamic equilibrium, the displacement field $\mathbf{u}(\mathbf{x}, t)$ generated by an impulsive point source satisfies the inhomogeneous wave equation~\eqref{eq:wave_inhomogeneous}.

% [Track E: Observational Grounding]
To measure this propagation across the regional network, continuous vertical telemetry was recorded on channel \texttt{EHZ} at \qty{100}{\hertz} (\cref{fig:record_section}).

% [Track E: Model Estimation]
Evaluating $N = 500$ arrivals with the non-linear locator (checkpoint \texttt{reloc_v1}), the estimated hypocentral depth was $\hat{z} = \qty{10.4}{\kilo\meter}$ ($95\%$ CI $[\qty{9.8}{\kilo\meter}, \qty{11.0}{\kilo\meter}]$).
```

