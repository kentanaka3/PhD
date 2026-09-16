# Publication-Grade TikZ and PGFPlots Standards

This reference governs reproducible vector diagrams, architectural schematics, and quantitative performance curves in scientific LaTeX documents.

---

## 1. Visual Evidence Boundaries in System Architectures

Every node in an architectural diagram must visually convey its implementation and verification status:

| Component Status | Border / Stroke Style | Fill Tone | Typographic Label Requirement |
| :--- | :--- | :--- | :--- |
| **Demonstrated / Implemented** | Solid `line width=0.8pt`, dark outline | Muted cool fill (`blue!8` or `teal!8`) | Module file path cited in caption or subtitle |
| **Persistence / Storage** | Solid double border (`double, double distance=1pt`) | Neutral slate fill (`gray!10`) | Storage protocol labeled |
| **Human / Expert Decision** | Hexagonal or chamfered border | Warm amber fill (`orange!10`) | Labeled ``Manual Review'' or ``Analyst Validation'' |
| **Planned / Proposed** | Dashed stroke (`dash pattern=on 3.5pt off 2.5pt`) | Muted patterned fill (`gray!4`) | Mandatory ``[Proposed]'' badge |

---

## 2. Production TikZ Architectural Diagram Template

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  >=Stealth,
  node distance=8mm and 10mm,
  every node/.style={font=\small},
  base/.style={rectangle, rounded corners=3pt, draw=black!80, line width=0.7pt, align=center, inner sep=5pt, minimum height=9mm},
  manual/.style={base, fill=orange!10, draw=orange!80!black},
  impl/.style={base, fill=blue!8, draw=blue!80!black, text width=30mm},
  audit/.style={base, fill=gray!10, draw=black!80, double, double distance=1pt, text width=30mm},
  planned/.style={base, fill=gray!4, draw=black!50, dash pattern=on 3.5pt off 2.5pt, text width=30mm, font=\small\itshape},
  arrow/.style={->, thick, draw=black!75},
  dashedarrow/.style={->, thick, dashed, draw=black!50}
]

  % Nodes
  \node[manual] (input) {Raw Waveform /\\Telemetry Stream};
  \node[impl, right=of input] (validate) {Signal Conditioning\\(\texttt{OGS/dsp/filter.py})};
  \node[audit, below=of validate] (storage) {Persistent Catalog\\(HDF5 Storage)};
  \node[impl, right=of validate] (compute) {Phase Inversion\\(\texttt{OGS/location})};
  \node[impl, above right=of compute] (outA) {Hypocenter Estimate ($\hat{\mathbf{x}}_e$)};
  \node[impl, below right=of compute] (outB) {Residual Field ($r_t$)};
  \node[planned, right=32mm of compute] (future) {[Proposed] Online\\Velocity Calibrator};

  % Connections
  \draw[arrow] (input) -- (validate);
  \draw[arrow] (validate) -- (storage);
  \draw[arrow] (validate) -- (compute);
  \draw[arrow] (compute) |- (outA);
  \draw[arrow] (compute) |- (outB);
  \draw[dashedarrow] (outA) -- (future);
  \draw[dashedarrow] (outB) -- (future);

  % Subsystem Bounding Box
  \begin{pgfonlayer}{background}
    \node[draw=blue!40, fill=blue!2, dashed, rounded corners=5pt, fit=(validate) (compute) (outA) (outB), inner sep=6pt] (subsys) {};
    \node[anchor=north west, font=\scriptsize\bfseries\color{blue!70!black}] at (subsys.north west) {Verified Seismic Subsystem};
  \end{pgfonlayer}

\end{tikzpicture}
\caption{System data processing and estimation pipeline. Solid blue boxes denote demonstrated, unit-tested software modules; double-bordered gray boxes represent persistent storage; the dashed box indicates the proposed online calibration module. Notice that signal conditioning occurs prior to state estimation.}
\label{fig:pipeline-architecture}
\end{figure}
```

---

## 3. Quantitative PGFPlots Standards

When plotting empirical curves, loss trajectories, or benchmark comparisons:
1. **Explicit Theoretical Baselines**: Plot analytical bounds or literature limits with labeled dashed rules using `extra y ticks`.
2. **Accessible Multi-Series Styling**: Vary both color and marker/stroke styles (e.g., solid blue with circles vs. dashed red with squares) for monochrome legibility.
3. **Explicit Units and Scales**: Label every axis with parameter symbols and SI units (e.g., `xlabel={Signal-to-Noise Ratio (SNR) [\unit{\decibel}]}`).
4. **Confidence Intervals**: When data is stochastic, plot standard deviation or $95\%$ confidence bounds using shaded error regions.

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
  \begin{axis}[
    width=0.88\linewidth,
    height=5.5cm,
    xlabel={Signal-to-Noise Ratio (SNR) [\unit{\decibel}]},
    ylabel={Estimation Error (RMSE) [\unit{\meter}]},
    xmin=0, xmax=25,
    ymin=0, ymax=10,
    xtick={0, 5, 10, 15, 20, 25},
    extra y ticks={1.5},
    extra y tick labels={Cram\'er--Rao Bound},
    extra y tick style={grid=major, grid style={dashed, red!70!black, line width=0.8pt}},
    legend pos=north east,
    legend cell align={left},
    font=\small,
    grid=both,
    grid style={dotted, gray!50}
  ]
    \addplot[thick, color=blue!80!black, mark=*] coordinates {
      (0, 8.8) (5, 6.2) (10, 3.7) (15, 2.1) (20, 1.7) (25, 1.55)
    };
    \addlegendentry{Proposed Estimator}

    \addplot[thick, color=black!60, dashed, mark=square*] coordinates {
      (0, 9.5) (5, 7.8) (10, 5.4) (15, 4.1) (20, 3.6) (25, 3.4)
    };
    \addlegendentry{Standard Baseline}
  \end{axis}
\end{tikzpicture}
\caption{Root-mean-square error (RMSE) versus signal-to-noise ratio (SNR) over $M=10^4$ Monte Carlo trials. The reader should notice the rapid convergence of the proposed estimator toward the theoretical Cram\'er--Rao lower bound (dashed red rule) above \qty{15}{\decibel}.}
\label{fig:snr-vs-rmse}
\end{figure}
```

---

## 4. Scientifically Explanatory 3-Part Caption Protocol

Every figure and table caption must satisfy the 3-part explanatory triad:
1. **Target & Scope**: Name the exact pipeline, model, dataset, or mathematical relationship depicted.
2. **Key Phenomenon ("Highlight")**: Explicitly articulate the primary takeaway, inflection point, or quantitative divergence the reader should observe.
3. **Evidence Anchor & Conditions**: Declare data origin, sample sizes ($N$), Monte Carlo iterations ($M$), baseline references, and operational constraints.

