# Publication-Grade Tabular Representation in Scientific LaTeX

This reference defines publication standards for scholarly tables, providing pre-calculated column ratio skeletons for `tabularx` that eliminate layout arithmetic errors during generation.

---

## 1. Foundational Tabular Typographic Rules

1. **Strict `booktabs` Triad**:
   - Use exclusively `\toprule` (opener), `\midrule` (header separator), `\bottomrule` (closer), and `\cmidrule(lr){start-end}` for grouped subheaders.
   - Use zero vertical rules. Ocular scanning is guided by clean typographic whitespace and alignment, never cell boundaries.
2. **Decimal Alignment via `siunitx`**:
   - Align numerical data columns using `S` column descriptors (e.g., `S[table-format=2.2]`, `S[table-format=3.1]`).
   - Guard non-numeric headers inside curly braces `{Header}` or `\multicolumn{1}{c}{\textbf{Header}}`.
   - Place physical units in column headers inside square brackets using `\unit{...}` (e.g., `Latency [\unit{\milli\second}]`). Keep data cells pure numbers.
3. **Self-Contained Notes via `threeparttable`**:
   - Enclose complex tables inside `threeparttable` with `\begin{tablenotes}[flushleft]`.
   - Explicitly define sample sizes ($N$), random seeds, confidence intervals ($95\%$ CI), and $p$-value significance thresholds ($*p < 0.05$).
4. **Mandatory 3-Part Caption Protocol**:
   Every table caption must satisfy:
   - **Scope**: Target system, dataset, or mathematical model evaluated.
   - **Highlight**: Key numerical phenomenon, optimal trade-off, or primary finding.
   - **Evidence**: Data source, test split, uncertainty metrics, and seed conditions.

---

## 2. Invariant Column Ratio Skeletons for `tabularx`

In `tabularx`, proportional column scaling is governed by `\hsize`:
$$
  w_i = k_i \cdot \bar{w}_X \quad \text{where} \quad \sum_{i=1}^{N_X} k_i = N_X.
$$
To avoid layout cramping or overfull margins, select directly from these four pre-calculated skeletons:

### Skeleton 1: Full-Width Numerical Benchmark (`tabular*`)
Use for quantitative performance evaluations, ablation studies, and error benchmarks:

```latex
\begin{table}[htbp]
  \centering
  \small
  \caption{Comparative performance evaluation across seismic phase-association benchmarks. Bold values denote optimal performance; underlined values indicate second-best baselines. All metrics report the mean $\pm$ standard deviation evaluated over five independent seeds on the curated test suite ($N=10^4$).}
  \label{tab:model-benchmarks}
  \begin{threeparttable}
    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l l S[table-format=2.2] S[table-format=2.2] S[table-format=3.1] c @{}}
      \toprule
      \textbf{Model / Architecture} & \textbf{Epistemic Status\tnote{a}} & {\textbf{Precision [\%]}} & {\textbf{Recall [\%]}} & {\textbf{Latency [\unit{\milli\second}]}} & {\textbf{Params [M]}} \\
      \midrule
      Linear Baseline               & Literature Baseline               & 74.20                     & 71.85                  & 1.2                                      & 0.05                   \\
      Random Forest ($T=500$)       & Literature Baseline               & 82.45                     & 81.10                  & 14.8                                     & 12.40                  \\
      ResNet-50 (Pretrained)        & Re-implemented                    & 89.60                     & 88.92                  & 28.5                                     & 25.56                  \\
      Transformer-Base              & Re-implemented                    & 92.15                     & 91.80                  & 45.2                                     & 86.40                  \\
      \textbf{Proposed Framework}   & \textbf{Demonstrated Work}        & \bfseries 95.40           & \bfseries 95.12        & 32.1                                     & 34.20                  \\
      Proposed (INT8 Quantized)     & Ablation Variant                  & \underline{94.85}         & \underline{94.50}      & \bfseries 9.4                            & \bfseries 8.55         \\
      \bottomrule
    \end{tabular*}
    \begin{tablenotes}[flushleft]
      \footnotesize
      \item[a] Status declares empirical verification: ``Literature Baseline'' denotes scores cited from published work; ``Demonstrated Work'' represents novel contributions verified by continuous integration tests.
    \end{tablenotes}
  \end{threeparttable}
\end{table}
```

---

### Skeleton 2: Two-Ratio Descriptive Table ($N_X = 2$)
Use for taxonomies, conceptual dictionaries, and trade-off registries ($k_1 = 0.60, k_2 = 1.40 \implies \sum = 2.00$):

```latex
\begin{table}[htbp]
  \centering
  \small
  \caption{Mathematical state vectors and physical governing domains for seismic hypocenter inversion. All coordinates refer to the WGS84 datum.}
  \label{tab:state-space-definitions}
  \begin{threeparttable}
    \begin{tabularx}{\linewidth}{
      >{\hsize=0.60\hsize\bfseries\raggedright\arraybackslash}X
      >{\hsize=1.40\hsize\raggedright\arraybackslash}X
    }
      \toprule
      \textbf{Mathematical Parameter} & \textbf{Governing Domain, Physical Unit, and Operational Definition} \\
      \midrule
      Hypocentral Coordinates $\mathbf{x}_e$ &
      $\mathbb{R}^3$ [\unit{\kilo\meter}]. Spatial coordinates $(\phi, \lambda, z)$ with focal depth $z \in [0, 700]$ constrained within crustal boundaries. \\
      \addlinespace[3pt]
      Origin Epoch $t_0$ &
      $\mathbb{R}_{\ge 0}$ [\unit{\second}]. Absolute temporal epoch measured relative to UTC midnight. \\
      \addlinespace[3pt]
      Velocity Structure $\mathbf{v}(z)$ &
      $L^\infty(\mathbb{R}^+)$ [\unit{\kilo\meter\per\second}]. Piecewise continuous P-wave velocity profile satisfying $v_p \ge \qty{1.5}{\kilo\meter\per\second}$. \\
      \addlinespace[3pt]
      Data Covariance $\mathbf{C}_d$ &
      $\mathbb{S}_{++}^M$ [\unit{\second\squared}]. Symmetric positive-definite matrix parameterizing pick arrival uncertainties across $M$ receiver channels. \\
      \bottomrule
    \end{tabularx}
    \begin{tablenotes}[flushleft]
      \footnotesize
      \item \textbf{Notation:} $\mathbb{S}_{++}^M$ denotes the convex cone of $M \times M$ symmetric positive-definite real matrices.
    \end{tablenotes}
  \end{threeparttable}
\end{table}
```

---

### Skeleton 3: Three-Ratio Comparative Table ($N_X = 3$)
Use for pipeline provenance, component audits, and comparative taxonomies ($k_1 = 0.50, k_2 = 0.80, k_3 = 1.70 \implies \sum = 3.00$):

```latex
\begin{table}[htbp]
  \centering
  \small
  \caption{Computational pipeline verification audit and evidence provenance matrix. Verified modules demonstrate zero-regression tests; planned modules indicate design specifications.}
  \label{tab:pipeline-provenance}
  \begin{threeparttable}
    \begin{tabularx}{\linewidth}{
      >{\hsize=0.50\hsize\bfseries\raggedright\arraybackslash}X
      >{\hsize=0.80\hsize\raggedright\arraybackslash}X
      >{\hsize=1.70\hsize\raggedright\arraybackslash}X
    }
      \toprule
      \textbf{Subsystem} & \textbf{Status} & \textbf{Source Module \& Verification Evidence Anchor} \\
      \midrule
      Waveform Ingestion &
      Implemented (Verified) &
      \texttt{OGS/pipeline/ingest.py}; verified against $10^4$ miniSEED records with automated checksums. \\
      \addlinespace[3pt]
      Phase Association &
      Implemented (Verified) &
      \texttt{OGS/association/spatio\_temporal.py}; verified against deterministic travel-time fixtures. \\
      \addlinespace[3pt]
      Online Relocation &
      Roadmap Specification &
      Specification in \texttt{doc/spec/streaming.tex}; Kalman formulation pending cluster integration. \\
      \bottomrule
    \end{tabularx}
    \begin{tablenotes}[flushleft]
      \footnotesize
      \item \textbf{Verification Standard:} All ``Implemented'' modules are anchored to continuous integration tests executing deterministic fixtures with tolerance $\epsilon < 10^{-6}$.
    \end{tablenotes}
  \end{threeparttable}
\end{table}
```

---

### Skeleton 4: System / Parameter Dictionary Table
Use for state space declarations, hyperparameter registries, and optimization bounds ($N_X = 2, k_1 = 0.70, k_2 = 1.30$ with fixed-width anchors):

```latex
\begin{table}[htbp]
  \centering
  \small
  \caption{Hyperparameter settings, search domains, and optimal configurations for deep phase picking.}
  \label{tab:hyperparameter-dictionary}
  \begin{threeparttable}
    \begin{tabularx}{\linewidth}{
      l
      >{\hsize=0.70\hsize\raggedright\arraybackslash}X
      c
      l
      >{\hsize=1.30\hsize\raggedright\arraybackslash}X
    }
      \toprule
      \textbf{Symbol} & \textbf{Hyperparameter} & \textbf{Domain} & \textbf{Optimal} & \textbf{Operational Search Bounds \& Rationale} \\
      \midrule
      $\eta$          & Learning Rate           & $\mathbb{R}_{> 0}$    & \num{1.0e-3}     & Log-uniform search over $[\num{1e-5}, \num{1e-2}]$ with AdamW optimizer. \\
      $B$             & Batch Size              & $\mathbb{N}$          & 128              & Evaluated over $\{32, 64, 128, 256\}$ constrained by GPU memory. \\
      $\lambda_{\mathrm{reg}}$ & Weight Decay   & $\mathbb{R}_{\ge 0}$  & \num{1.0e-4}     & Decoupled weight regularization to prevent high-frequency noise fitting. \\
      \bottomrule
    \end{tabularx}
    \begin{tablenotes}[flushleft]
      \footnotesize
      \item \textbf{Protocol:} Optimization executed across five independent runs with cross-entropy loss.
    \end{tablenotes}
  \end{threeparttable}
\end{table}
```

