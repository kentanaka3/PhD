# Semantic Cross-Referencing (`cleveref`) and Acronym Management (`acro`)

This reference defines positive semantic routing for cross-referencing and acronym expansion in scientific LaTeX documents.

---

## 1. Modern Semantic Cross-Referencing Architecture (`cleveref`)

The `cleveref` package intercepts counter increments to record entity types, separating cross-referencing intent from typographic presentation.

### A. Preamble Loading Sequence
Maintain this strict package sequence to ensure valid hypertarget registration:
```latex
% 1. Math engines
\usepackage{amsmath, mathtools, amsthm}

% 2. Floats & Tables
\usepackage{graphicx, booktabs, caption, subcaption, algorithm, algpseudocode}

% 3. Hyperlinks (loaded second-to-last)
\usepackage[colorlinks=true, linkcolor=blue!75!black, citecolor=teal!75!black]{hyperref}

% 4. Semantic cross-referencing (strictly last)
\usepackage{cleveref}

% Format configurations
\crefname{equation}{Eq.}{Eqs.}
\Crefname{equation}{Equation}{Equations}
\creflabelformat{equation}{(#2#1#3)}
\crefname{figure}{Figure}{Figures}
\Crefname{figure}{Figure}{Figures}
\crefname{table}{Table}{Tables}
\Crefname{table}{Table}{Tables}
\crefname{section}{Section}{Sections}
\Crefname{section}{Section}{Sections}
\crefname{chapter}{Chapter}{Chapters}
\Crefname{chapter}{Chapter}{Chapters}
\crefname{theorem}{Theorem}{Theorems}
\Crefname{theorem}{Theorem}{Theorems}
```

### B. Canonical Semantic Cross-Referencing Dispatch
| Entity | Discourse Context | Canonical Macro | Rendered Output |
| :--- | :--- | :--- | :--- |
| **Equation** | Mid-sentence | `\cref{eq:navier}` | Eq.~(2.4) |
| **Equation** | Sentence-initial | `\Cref{eq:navier}` | Equation (2.4) |
| **Equation** | Noun present in prose | `the elastodynamic equation~\eqref{eq:navier}` | the elastodynamic equation (2.4) |
| **Equation** | Pairwise reference | `\cref{eq:p_wave,eq:s_wave}` | Eqs.~(2.4) and (2.5) |
| **Equation** | Contiguous range | `\cref{eq:1,eq:2,eq:3}` | Eqs.~(2.4) to (2.6) |
| **Figure** | Mid-sentence | `\cref{fig:pipeline}` | Figure 3.1 |
| **Figure** | Sentence-initial | `\Cref{fig:pipeline}` | Figure 3.1 |
| **Figure** | Range / Plural | `\cref{fig:a,fig:b}` | Figures 3.1 and 3.2 |
| **Subfigure**| Autonomous | `\cref{fig:panel_a}` | Figure 3.1(a) |
| **Table** | Mid-sentence | `\cref{tab:metrics}` | Table 4.2 |
| **Section** | Mid-sentence | `\cref{sec:methods}` | Section 3 |
| **Chapter** | Mid-sentence | `\cref{chap:clustering}` | Chapter 5 |
| **Theorem** | Mid-sentence | `\cref{thm:convergence}` | Theorem 4.1 |

### C. The Five Authoring Invariants
1. **Autonomous Macro**: Emit exclusively `\cref{key}` mid-sentence. Do not write manual nouns (`Figure~\ref` or `Eq.~\cref`).
2. **Sentence-Initial**: Emit `\Cref{key}` when starting a sentence to produce full capitalized words (*"Equation (1)"*, *"Figure 2"*).
3. **Qualified Noun**: When the noun phrase is already part of the sentence grammar, emit strictly `\eqref{key}` to supply parenthesized numbers without repeating the noun.
4. **Compound Argument**: Pass multiple keys as comma-separated lists inside a single call (`\cref{k1,k2,k3}`) to activate automatic conjunctions and range compression.
5. **Clean Parentheticals**: Embed `\cref{key}` directly inside parenthetical remarks: `(derived in \cref{eq:wave}; cf.~\eqref{eq:aux})`.

---

## 2. Acronym Management via Positive Grammatical Role Dispatch (`acro` v3)

Configure `acro` in the preamble to manage parenthetical expansions and reset lifecycles across chapters:

```latex
\usepackage{acro}
\acsetup{
  cite/group     = true,
  cite/group/cmd = \cite,
  cite/cmd       = \parencite,
  make-links     = true
}
\AddToHook{cmd/chapter/before}{\acresetall}
```

Declare acronyms systematically:
```latex
\DeclareAcronym{ETAS}{
  short = ETAS,
  long  = Epidemic-Type Aftershock Sequence,
  cite  = ogata1988
}
\DeclareAcronym{PINN}{
  short = PINN,
  long  = Physics-Informed Neural Network,
  cite  = raissi2019
}
```

### Positive Grammatical Role Dispatch Framework
Select the macro matching the syntactic function of the term:

| Syntactic Function | Recommended Macro | Rendered Output (First Call) | Subsequent Calls |
| :--- | :--- | :--- | :--- |
| **Role A: Standard Intro in Prose** | `\ac{KEY}` | Long Form (SHORT, Citation) | SHORT |
| **Role B: Grammatical Subject** | `\acl{KEY} (\acs{KEY})` | Long Form (SHORT) | Long Form (SHORT) |
| **Role B: Adjective / Modifier** | `\acl{KEY}` | Long Form | Long Form |
| **Role C: Section Title / Caption** | `\acs{KEY}` | SHORT | SHORT |
| **Role D: Chapter Re-Introduction** | `\acf{KEY}` | Long Form (SHORT, Citation) | Long Form (SHORT, Citation) |
| **Role E: Plural Entities** | `\acp{KEY}` | Long Forms (SHORTs, Citation) | SHORTs |

### Examples
- **Role A (Standard Running Text)**:
  ```latex
  We model clustering using an \ac{ETAS} point process.
  ```
  *Renders:* We model clustering using an Epidemic-Type Aftershock Sequence (ETAS, Ogata, 1988) point process.

- **Role B (Grammatical Subject / Direct Noun Phrase)**:
  ```latex
  The \acl{PINN} (\acs{PINN}) architecture embeds differential operators into the loss function.
  ```
  *Renders:* The Physics-Informed Neural Network (PINN) architecture embeds differential operators into the loss function.

- **Role C (Headings and Captions)**:
  ```latex
  \subsection{Convergence of the \acs{PINN} Estimator}
  \caption{Validation loss comparison for the \acs{PINN} baseline.}
  ```
  *Renders clean short forms without polluting the Table of Contents with citations.*

