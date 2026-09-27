# Mathematical, Physical, and Computational Exposition: Formal Rigor and Notation Standards

This reference governs mathematical, theoretical physics, and computational typesetting in scientific LaTeX documents. It ensures strict domain bounding, consistent operator typography, and modern delimiter discipline. Consistent examples and explicit intent can improve output reliability, model-dependently.

---

## 1. Environment Selection Hierarchy

Adhere to modern AMS-LaTeX (`amsmath`, `mathtools`) standards. Choose the mathematical environment based on whether the formula represents a numbered evidentiary anchor or an unnumbered intermediate transformation:

| Mathematical Context | Environment | Syntax Pattern | Typographic Purpose |
| :--- | :--- | :--- | :--- |
| **Numbered Theoretical Anchor** | `equation` | `\begin{equation} ... \label{eq:key} \end{equation}` | Primary governing equations, laws, or theorems referenced later in the text. |
| **Unnumbered Intermediate Step** | `$$ ... $$` | `$$ ... $$` | Single-line algebraic transformations, definitions, or substitutions not referenced elsewhere. |
| **Multi-Line Numbered Derivation** | `align` | `\begin{align} ... \label{eq:a} \\ ... \label{eq:b} \end{align}` | Multi-line systems or derivations where individual lines require independent cross-reference. |
| **Multi-Line Unnumbered Derivation** | `align*` | `\begin{align*} ... \\ ... \end{align*}` | Multi-line algebraic reductions or proofs where line-by-line numbering is unnecessary. |
| **Multi-Line Single-Number Formula** | `equation` + `aligned` | `\begin{equation} \begin{aligned} ... \end{aligned} \label{eq:key} \end{equation}` | Long multi-line expression or derivation that shares a single equation number. |
| **Grouped Independent Equations** | `gather` | `\begin{gather} ... \label{eq:a} \\ ... \label{eq:b} \end{gather}` | Consecutive formulas centered independently without horizontal relation-symbol alignment. |
| **Long Broken Formula** | `multline` | `\begin{multline} ... \\ ... \end{multline}` | Single long expression exceeding text width, broken across lines without column alignment. |

> [!TIP]
> **Numbering Discipline**: Reserve equation numbers for formulas that are cross-referenced or represent key milestones. Use `$$ ... $$` or `align*` for intermediate steps to avoid label proliferation.

---

## 2. Semantic Mathematical and Physical Typefaces

Rigorous scientific exposition demands unambiguous visual separation between scalar fields, coordinate vectors, discrete matrices, constitutive tensors, abstract algebraic structures, computational complexity classes, algorithm specifications, program semantics, and descriptive annotations. Adhere strictly to ISO 80000-2, AMS-LaTeX, ACM/IEEE, and IUPAP/SIAM typographic conventions.

### Master Typeface Discipline Matrix

| Font Family / Macro | Semantic Scope & Mathematical Entity | LaTeX Syntax | Canonical Example | Disambiguation & Typographic Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Math Italic** (Default) | Scalar variables, real coordinates, running indices, scalar fields | `$x, y, z, t, m, \rho, \theta$` | $f(x, t) = A \sin(k x - \omega t)$ | Distinguishes continuous mathematical variables and running counters from fixed constants and labels. |
| **Upright Roman** (`\mathrm`) | Invariant mathematical constants | `\mathrm{e}, \mathrm{i}, \uppi` | $\mathrm{e}^{\mathrm{i}\pi} + 1 = 0$ | Prevents confusion with variable charge/strain $e$ or spatial coordinate index $i$. |
| | Differential & variational operators | `\mathrm{d}, \delta` | $\int_\Omega f(\mathbf{x}) \,\mathrm{d}^3\mathbf{x}$ | Typeset with preceding thin space `\,`; denotes the exterior derivative / measure, not a variable product $d \cdot x$. |
| | Fixed textual subscripts & labels | `\mathrm{kin}, \mathrm{eff}, \mathrm{arr}, \max` | $E_{\mathrm{kin}}, v_{\mathrm{P}}, v_{\mathrm{S}}, T_{\mathrm{eff}}, \lambda_{\max}$ | Distinguishes qualitative descriptive abbreviations from running variable indices ($a_i$ vs $a_{\mathrm{i}}$). |
| | Standard multi-letter functions | `\sin, \exp, \ln, \operatorname{erf}` | $\operatorname{erf}(x), \operatorname{sinc}(\omega)$ | Standard functions must be typeset upright, never as consecutive italic variables $s \cdot i \cdot n$. |
| | Physical SI units & dimensions | `\unit{\kilo\meter}, \unit{\pascal}` | $\qty{15}{\kilo\meter\per\second}$ | Physical units are never italicized; always enforce thin non-breaking space between quantity and unit. |
| | Floating-point precision formats | `\mathrm{FP32}, \mathrm{BF16}` | $\mathrm{FP32}, \mathrm{BF16}, \mathrm{INT8}$ | Upright font isolates IEEE numerical data formats from algebraic variables. |
| | Chemical elements & particles | `\mathrm{Fe}, \mathrm{e}^-, \mathrm{p}^+` | $^{238}_{\phantom{0}92}\mathrm{U} \to \alpha + \dots$ | Chemical and particle species symbols remain strictly upright roman. |
| | Lie groups & matrix groups | `\mathrm{SO}(3), \mathrm{SU}(N), \mathrm{SE}(3)` | $R \in \mathrm{SO}(3), U \in \mathrm{SU}(2)$ | Continuous Lie groups are upright; distinguishes the group manifold from individual algebra elements. |
| **Bold Upright Roman** (`\mathbf`) | Coordinate spatial vectors & displacements | `\mathbf{x}, \mathbf{u}, \mathbf{v}, \mathbf{k}, \mathbf{f}` | $\mathbf{u}(\mathbf{x}, t) \in \mathbb{R}^3$ | Rank-1 geometric entities in Euclidean space. |
| | Coordinate matrices & 2nd-order tensors | `\mathbf{A}, \mathbf{K}, \mathbf{M}, \mathbf{C}, \mathbf{I}` | $\mathbf{M}\ddot{\mathbf{u}} + \mathbf{C}\dot{\mathbf{u}} + \mathbf{K}\mathbf{u} = \mathbf{f}$ | Rank-2 discrete linear operators, mass/stiffness matrices, and second-order identity $\mathbf{I}$. |
| | Neural network layer weight matrices | `\mathbf{W}^{(l)}, \mathbf{b}^{(l)}` | $\mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})$ | Distinguishes layer projection operators from abstract mathematical mappings. |
| | Graph adjacency & Laplacian matrices | `\mathbf{A}, \mathbf{D}, \mathbf{L}` | $\mathbf{L} = \mathbf{D} - \mathbf{A}$ | Discrete graph matrices distinct from continuous operators. |
| **Bold Italic** (`\bm` / `\boldsymbol`) | Greek spatial vectors & parameters | `\bm{\omega}, \bm{\xi}, \bm{\theta}, \bm{\mu}` | $\bm{\omega} = \nabla \times \mathbf{u}, \bm{\theta} \in \mathbb{R}^p$ | Mandatory for Greek vector quantities (where `\mathbf` produces undefined or upright glyphs). |
| | Greek stress & strain tensors | `\bm{\sigma}, \bm{\varepsilon}, \bm{\tau}` | $\bm{\sigma} = \lambda \operatorname{Tr}(\bm{\varepsilon})\mathbf{I} + 2\mu\bm{\varepsilon}$ | Continuum mechanics second-order symmetric stress/strain fields. |
| | Bold italic vectors in Sobolev spaces | `\bm{u} \in H^1_0(\Omega)^d` | $(\bm{u}, \bm{v})_{L^2(\Omega)}$ | Common in modern PDE analysis to distinguish vector fields from scalar test functions $v$. |
| **Bold Sans-Serif** (`\bm{\mathsf{...}}` / `\mathbfsf`) | 4th-order constitutive & elasticity tensors | `\bm{\mathsf{C}}, \bm{\mathsf{S}}, \bm{\mathsf{I}}^{\mathrm{sym}}` | $\bm{\sigma} = \bm{\mathsf{C}} : \bm{\varepsilon}$ | Visually separates rank-4 stiffness/compliance tensors from rank-2 matrices $\mathbf{C}$ and scalar constants $C$. |
| | Random vectors in signal processing | `\bm{\mathsf{X}}, \bm{\mathsf{Y}}` | $\bm{\mathsf{Y}} = \mathbf{H}\bm{\mathsf{X}} + \bm{\mathsf{W}}$ | Multidimensional stochastic processes distinct from deterministic sample vectors. |
| **Sans-Serif** (`\mathsf`) | Random variables (modern probability) | `\mathsf{X}, \mathsf{Y}, \mathsf{Z}` | $\mathsf{X} \sim \mathcal{N}(\mu, \sigma^2)$ | Distinguishes stochastic variables $\mathsf{X}$ from their deterministic scalar realizations $x \in \mathbb{R}$. |
| | Graph-theoretical objects & trees | `\mathsf{G} = (\mathsf{V}, \mathsf{E})` | $\mathsf{e} = (u, v) \in \mathsf{E}$ | Distinguishes graphs, vertices, and edges from continuum field domains $\Omega$ and sets $V$. |
| | Computational complexity classes | `\mathsf{P}, \mathsf{NP}, \mathsf{BQP}, \mathsf{NC}^k` | $\mathsf{NP}\text{-complete}, \mathsf{PH}$ | Standard complexity-theoretic typography across theoretical computer science. |
| | Type names in programming semantics | `\mathsf{Nat}, \mathsf{Bool}, \mathsf{Unit}` | $\Gamma \vdash e : \mathsf{Bool}$ | Distinguishes program types from mathematical variables and sets. |
| | Dimensional symbols (Dimensional analysis) | `\mathsf{M}, \mathsf{L}, \mathsf{T}, \mathsf{\Theta}` | $[\text{force}] = \mathsf{M}\mathsf{L}\mathsf{T}^{-2}$ | ISO dimensional quantities denoting Mass, Length, Time, Temperature. |
| **Small Caps** (`\textsc`) | Computational decision problems | `\textsc{3-Sat}, \textsc{MaxCut}` | $\textsc{3-Sat} \le_{\mathrm{P}} \textsc{VertexCover}$ | Canonical typography for formal decision and search problems. |
| | Named algorithms and model baselines | `\textsc{Transformer}, \textsc{PAk}` | $\textsc{Quicksort}, \textsc{ResNet}$ | Isolates algorithmic designs from generic text descriptions. |
| **Monospace / Typewriter** (`\texttt`) | Source code identifiers, files, schemas | `\texttt{filter.py}, \texttt{lr=1e-4}` | `\texttt{reloc_v1}`, `\texttt{float32}` | Deterministic code-level identifiers, config keys, and paths. |
| **Blackboard Bold** (`\mathbb`) | Number fields, division rings, lattices | `\mathbb{N}, \mathbb{Z}, \mathbb{Q}, \mathbb{R}, \mathbb{C}, \mathbb{H}` | $\mathbf{x} \in \mathbb{R}^d, z \in \mathbb{C}$ | Universal sets of numbers, algebraic fields, and quaternions $\mathbb{H}$. |
| | Boolean domain & finite fields | `\mathbb{B}, \mathbb{F}_q, \mathbb{F}_2` | $\mathbb{B} = \{0, 1\}, \mathbb{F}_p$ | Discrete algebraic fields and digital bit domains. |
| | Probability & mathematical expectation | `\mathbb{P}, \mathbb{E}, \mathbb{V}\mathrm{ar}` | $\mathbb{E}[f(\mathsf{X})] = \int f(x) \,\mathrm{d}\mathbb{P}$ | Distinguishes the expectation functional $\mathbb{E}$ from total energy $E$ or Young's modulus $E$. |
| | Projective & affine spaces | `\mathbb{P}^n, \mathbb{A}^n, \mathbb{K}` | $[x_0 : \dots : x_n] \in \mathbb{P}^n(\mathbb{R})$ | Geometric projective spaces. |
| | Indicator functions | `\mathbb{I}_A` or `\mathds{1}_A` | $\mathbb{I}_A(x) = [x \in A]$ | Characteristic/indicator measure on measurable subsets. |
| **Calligraphic & Script** (`\mathcal`, `\mathscr`) | Higher-order data tensors (Computer Vision) | `\bm{\mathcal{X}} \in \mathbb{R}^{B \times C \times H \times W}` | $\bm{\mathcal{Y}} = \bm{\mathcal{X}} \times_n \mathbf{U}$ | Distinguishes rank-$\ge 3$ multidimensional tensors from matrices and vectors. |
| | Hilbert, Banach, and Sobolev function spaces | `\mathcal{H}, \mathcal{B}, \mathcal{W}^{k,p}` | $\mathcal{H} = L^2(\Omega), u \in \mathcal{H}$ | Infinite-dimensional topological vector spaces. |
| | Test function spaces & distributions | `\mathcal{D}(\Omega), \mathcal{S}(\mathbb{R}^d)` | $\phi \in \mathcal{D}(\Omega), T \in \mathcal{D}'(\Omega)$ | Test spaces and Schwartz space of rapidly decreasing functions. |
| | Lagrangian & Hamiltonian densities | `\mathcal{L}, \mathcal{H}` | $S = \int \mathcal{L}(\phi, \partial_\mu\phi) \,\mathrm{d}^4x$ | Distinguishes volumetric densities $\mathcal{L}, \mathcal{H}$ from integrated action/Hamiltonian $L, H$. |
| | Statistical $\sigma$-algebras & sample spaces | `\mathcal{F}, \mathcal{B}(\mathbb{R}^d)` | $(\Omega, \mathcal{F}, \mathbb{P})$ | Measurable spaces and Borel $\sigma$-algebras. |
| | Machine learning datasets & hypothesis classes | `\mathcal{D}, \mathcal{M}, \mathcal{H}` | $\mathcal{D}_{\mathrm{train}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ | Training/validation corpora and hypothesis function sets. |
| | Asymptotic Landau upper bound (Big-O) | `\mathcal{O}(n \log n)` | $T(n) = \mathcal{O}(n^2)$ | Distinguishes complexity bounds from scalar variables $O$ or output matrices. |
| | Integral transform operators | `\mathcal{F}, \mathcal{L}, \mathcal{H}` | $\mathcal{F}\{f\}(\omega) = \hat{f}(\omega)$ | Continuous Fourier, Laplace, and Hilbert transform operators. |
| **Fraktur / Gothic** (`\mathfrak`) | Rademacher complexity (Learning theory) | `\mathfrak{R}_n(\mathcal{H}), \widehat{\mathfrak{R}}_S` | $\widehat{\mathfrak{R}}_S(\mathcal{H}) = \mathbb{E}_{\bm{\sigma}}[\dots]$ | Empirical and distributional Rademacher complexities in statistical learning. |
| | Lie algebras (tangent spaces of Lie groups) | `\mathfrak{g}, \mathfrak{su}(n), \mathfrak{so}(3), \mathfrak{se}(3)` | $[X, Y] \in \mathfrak{so}(3)$ | Distinguishes the infinitesimal generator algebra $\mathfrak{g}$ from the group manifold $G$. |
| | Ideals in commutative algebra | `\mathfrak{p}, \mathfrak{q}, \mathfrak{m}` | $\mathfrak{p} \subset R, \mathfrak{m} \in \operatorname{MaxSpec}(R)$ | Prime, primary, and maximal ideals in ring theory. |
| | Cardinality of the continuum | `\mathfrak{c}` | $\mathfrak{c} = 2^{\aleph_0} = \|\mathbb{R}\|$ | Transfinite set theory and cardinal numbers. |

---

### A. Coordinate Vectors, Matrices, and Higher-Order Constitutive Tensors
Maintain strict structural hierarchy according to tensor rank:
- **Rank-0 (Scalars)**: Lightface italic ($s, \rho, \mu, \lambda, \theta$).
- **Rank-1 (Vectors)**: Bold Roman lowercase ($\mathbf{x}, \mathbf{u}, \mathbf{v}, \mathbf{k}, \mathbf{f}$) or bold italic Greek ($\bm{\omega}, \bm{\xi}, \bm{\theta}$).
- **Rank-2 (Matrices & Second-Order Tensors)**: Bold Roman uppercase ($\mathbf{A}, \mathbf{K}, \mathbf{M}, \mathbf{C}, \mathbf{I}$) or bold Greek ($\bm{\sigma}, \bm{\varepsilon}, \bm{\tau}$).
- **Rank-4 (Constitutive & Elasticity Tensors)**: Bold Sans-Serif uppercase ($\bm{\mathsf{C}}, \bm{\mathsf{S}}, \bm{\mathsf{I}}^{\mathrm{sym}}$):
  $$
    \bm{\sigma} = \bm{\mathsf{C}} : \bm{\varepsilon} \iff \sigma_{ij} = C_{ijkl} \varepsilon_{kl}.
  $$
  For an isotropic linear elastic medium:
  $$
    C_{ijkl} = \lambda \delta_{ij} \delta_{kl} + \mu \left( \delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk} \right), \quad \bm{\mathsf{I}}^{\mathrm{sym}}_{ijkl} = \frac{1}{2}\left( \delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk} \right).
  $$
  Compliance relation: $\bm{\varepsilon} = \bm{\mathsf{S}} : \bm{\sigma}$ where $\bm{\mathsf{S}} = \bm{\mathsf{C}}^{-1}$.
- **Voigt Notation Vectorization**: Contract symmetric $3 \times 3$ stress and strain tensors into 6-dimensional algebraic vectors:
  $$
    \vec{\bm{\sigma}}_{\mathrm{V}} = \begin{bmatrix} \sigma_{xx} & \sigma_{yy} & \sigma_{zz} & \sigma_{yz} & \sigma_{xz} & \sigma_{xy} \end{bmatrix}^\top, \quad
    \vec{\bm{\varepsilon}}_{\mathrm{V}} = \begin{bmatrix} \varepsilon_{xx} & \varepsilon_{yy} & \varepsilon_{zz} & 2\varepsilon_{yz} & 2\varepsilon_{xz} & 2\varepsilon_{xy} \end{bmatrix}^\top.
  $$
- **Tensor Products & Contraction Operations**:
  - Dyadic (Outer) Product: $\mathbf{a} \otimes \mathbf{b} \in \mathbb{R}^{d \times d}$, where $(\mathbf{a} \otimes \mathbf{b})_{ij} = a_i b_j$.
  - Euclidean Inner Product: $\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^\top \mathbf{b} = \sum_{i=1}^d a_i b_i$.
  - Double-Dot (Frobenius) Contraction: $\mathbf{A} : \mathbf{B} = \operatorname{Tr}(\mathbf{A}^\top \mathbf{B}) = A_{ij} B_{ij}$.
- **Domain Bounding**: Formally bound every variable's domain upon first introduction:
  $$
    \mathbf{u} \in H^1_0(\Omega)^d, \quad \bm{\sigma} \in L^2(\Omega)_{\mathrm{sym}}^{d \times d}, \quad \bm{\theta} \in \Theta \subset \mathbb{R}^p, \quad t \in [0, T], \quad \mathbf{x} \in \Omega \subset \mathbb{R}^3.
  $$

---

### B. Tensor Calculus, Differential Forms, and Exterior Algebra
In continuum mechanics, general relativity, and field theory, express tensors in component index notation with consistent covariant (lower) and contravariant (upper) indices, or in coordinate-free differential forms:

- **Component Index Notation (Einstein Summation)**:
  - Contravariant Components: $T^{\mu\nu}, v^\alpha, \mathrm{d}x^\mu$.
  - Covariant Metric & Forms: $g_{\mu\nu}, \omega_\alpha, \partial_{x^\mu} \equiv \frac{\partial}{\partial x^\mu}$.
  - Horizontal Index Staggering: Preserve contraction order during metric raising and lowering:
    $$
      T^{\mu}_{\phantom{\mu}\nu} = g_{\nu\alpha} T^{\mu\alpha}, \quad T_{\mu}^{\phantom{\mu}\nu} = g_{\mu\alpha} T^{\alpha\nu}, \quad R^{\rho}_{\phantom{\rho}\sigma\mu\nu}.
    $$
  - Christoffel Symbols: $\Gamma^\lambda_{\mu\nu} = \frac{1}{2} g^{\lambda\sigma}\left( \partial_{x^\mu} g_{\nu\sigma} + \partial_{x^\nu} g_{\mu\sigma} - \partial_{x^\sigma} g_{\mu\nu} \right)$.
  - Covariant Derivatives: $\nabla_\mu v^\alpha = \partial_{x^\mu} v^\alpha + \Gamma^\alpha_{\mu\beta} v^\beta, \quad \nabla_\mu \omega_\nu = \partial_{x^\mu} \omega_\nu - \Gamma^\alpha_{\mu\nu} \omega_\alpha$.
  - Cauchy Stress & Strain: $\sigma_{ij} = \lambda \delta_{ij} \varepsilon_{kk} + 2\mu \varepsilon_{ij}, \quad \varepsilon_{ij} = \frac{1}{2}\left( \partial_{x_j} u_i + \partial_{x_i} u_j \right)$.

- **Coordinate-Free Exterior Calculus**:
  - Differential $k$-Forms: $\omega \in \Omega^k(M)$.
  - Wedge Product (Graded Anticommutative): For $\alpha \in \Omega^p(M)$ and $\beta \in \Omega^q(M)$:
    $$
      \alpha \wedge \beta = (-1)^{pq} \beta \wedge \alpha.
    $$
  - Exterior Derivative: Nilpotent coboundary operator $\mathrm{d} \colon \Omega^k(M) \to \Omega^{k+1}(M)$ with $\mathrm{d}^2 = \mathrm{d} \circ \mathrm{d} = 0$.
  - Hodge Star Dual: Metric isomorphism $\star \colon \Omega^k(M) \to \Omega^{n-k}(M)$.
  - Interior Product (Contraction with vector field $X$): $\iota_X \omega \equiv X \mathbin{\lrcorner} \omega \in \Omega^{k-1}(M)$.
  - Lie Derivative & Cartan's Magic Formula:
    $$
      \mathcal{L}_X \omega = \mathrm{d}(\iota_X \omega) + \iota_X(\mathrm{d}\omega).
    $$
  - Codifferential: $\delta = (-1)^{n(k-1)+1} s \star \mathrm{d} \star \colon \Omega^k(M) \to \Omega^{k-1}(M)$ (where $s = \operatorname{sgn}(\det g)$).
  - Hodge-de Rham Laplacian: $\Delta = \mathrm{d}\delta + \delta\mathrm{d}$.

---

### C. Lie Groups, Lie Algebras, and Gauge Symmetries
- **Continuous Lie Groups**: Typeset in upright Roman capitals:
  $$
    G \in \{\mathrm{SO}(3), \mathrm{SU}(N), \mathrm{SE}(3), \mathrm{GL}(n, \mathbb{R}), \mathrm{Sp}(2n, \mathbb{R})\}.
  $$
- **Lie Algebras**: Typeset in lowercase Fraktur corresponding to the group:
  $$
    \mathfrak{g} \in \{\mathfrak{so}(3), \mathfrak{su}(N), \mathfrak{se}(3), \mathfrak{gl}(n, \mathbb{R}), \mathfrak{sp}(2n, \mathbb{R})\}.
  $$
- **Lie Bracket**: Bilinear, alternating, satisfying the Jacobi identity:
  $$
    [\cdot, \cdot] \colon \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g}, \quad [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0.
  $$
- **Adjoint Representations**:
  - Group Adjoint: $\operatorname{Ad}_g \colon G \to \operatorname{Aut}(\mathfrak{g})$, where $\operatorname{Ad}_g(X) = g X g^{-1}$.
  - Algebra Adjoint: $\operatorname{ad}_X \colon \mathfrak{g} \to \operatorname{End}(\mathfrak{g})$, where $\operatorname{ad}_X(Y) = [X, Y]$.
  - Killing Metric: $\kappa(X, Y) = \operatorname{Tr}(\operatorname{ad}_X \circ \operatorname{ad}_Y)$.
- **Exponential Map**: $\exp \colon \mathfrak{g} \to G$ via $\exp(tX) = \sum_{k=0}^\infty \frac{t^k}{k!} X^k$.
- **Gauge Potential & Curvature 2-Form**:
  - Gauge Connection (Vector Potential): $A = A_\mu^a T_a \,\mathrm{d}x^\mu \in \Omega^1(M; \mathfrak{g})$, where $\{T_a\}$ are Lie algebra generators with $[T_a, T_b] = f_{ab}^{\phantom{ab}c} T_c$.
  - Field Strength (Curvature 2-Form):
    $$
      F = \mathrm{d}A + A \wedge A = \frac{1}{2} F_{\mu\nu}^a T_a \,\mathrm{d}x^\mu \wedge \mathrm{d}x^\nu, \quad F_{\mu\nu}^a = \partial_{x^\mu} A_\nu^a - \partial_{x^\nu} A_\mu^a + f_{bc}^{\phantom{bc}a} A_\mu^b A_\nu^c.
    $$

---

### D. Vector Calculus, Differential Operators, and Kinematic Rates
- **Nabla / Del Operators**: Use standard $\nabla$ with explicit vector products:
  - Gradient: $\nabla \phi$ or $\operatorname{grad} \phi$.
  - Divergence: $\nabla \cdot \mathbf{u}$ or $\operatorname{div} \mathbf{u}$.
  - Curl / Rotor: $\nabla \times \mathbf{u}$ or $\operatorname{curl} \mathbf{u}$.
  - Scalar Laplacian: $\nabla^2 \phi$ or $\Delta \phi$.
  - Vector Laplacian: $\nabla^2 \mathbf{u} = \nabla(\nabla \cdot \mathbf{u}) - \nabla \times (\nabla \times \mathbf{u})$.
- **Differentials**: Typeset the differential operator in upright roman (`\mathrm{d}`) with a thin leading space (`\,`):
  $$
    \int_{\Omega} f(\mathbf{x}) \,\mathrm{d}^3\mathbf{x}, \quad \oint_{\partial\Omega} \mathbf{F} \cdot \mathbf{n} \,\mathrm{d}S, \quad \frac{\mathrm{d}y}{\mathrm{d}t}.
  $$
- **Kinematic Rates of Change**:
  - Total Time Derivative (Newton Fluxion): $\dot{\mathbf{u}} = \frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t}, \quad \ddot{\mathbf{u}} = \frac{\mathrm{d}^2\mathbf{u}}{\mathrm{d}t^2}$.
  - Material / Convective Derivative (Continuum mechanics):
    $$
      \frac{\mathrm{D}\phi}{\mathrm{D}t} = \frac{\partial \phi}{\partial t} + (\mathbf{u} \cdot \nabla)\phi, \quad
      \frac{\mathrm{D}\mathbf{v}}{\mathrm{D}t} = \frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v}.
    $$
  - Strain-Rate and Spin Tensors:
    $$
      \dot{\bm{\varepsilon}} = \frac{1}{2}\left( \nabla \mathbf{v} + (\nabla \mathbf{v})^\top \right), \quad \bm{\omega} = \frac{1}{2}\left( \nabla \mathbf{v} - (\nabla \mathbf{v})^\top \right).
    $$

---

### E. Probability, Stochastic Processes, and Information Geometry
- **Random Variables and Realizations**:
  - Stochastic variables: Sans-serif uppercase $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ (or uppercase serif $X, Y, Z$).
  - Deterministic sample realizations: Lowercase lightface italic $x \in \mathcal{X}, y \in \mathcal{Y}$.
- **Probability Measure, Expectation, and Covariance**:
  $$
    \mathbb{P}(E) = \int_E \mathrm{d}\mathbb{P}, \quad \mathbb{E}[\mathsf{X}] = \int_\Omega \mathsf{X}(\omega) \,\mathrm{d}\mathbb{P}(\omega), \quad \operatorname{Var}(\mathsf{X}) = \mathbb{E}\bigl[(\mathsf{X} - \mathbb{E}[\mathsf{X}])^2\bigr].
  $$
  Covariance matrix for a multivariate random vector $\bm{\mathsf{X}} \in \mathbb{R}^d$:
  $$
    \bm{\Sigma} = \operatorname{Cov}(\bm{\mathsf{X}}) = \mathbb{E}\bigl[(\bm{\mathsf{X}} - \bm{\mu})(\bm{\mathsf{X}} - \bm{\mu})^\top\bigr] \in \mathbb{R}_{\ge 0}^{d \times d}.
  $$
- **Stochastic Differential Equations (Itô Calculus)**:
  $$
    \mathrm{d}\mathsf{X}_t = \bm{\mu}(\mathsf{X}_t, t)\,\mathrm{d}t + \bm{\sigma}(\mathsf{X}_t, t)\,\mathrm{d}\mathsf{W}_t, \quad (\mathrm{d}\mathsf{W}_t)^2 = \mathrm{d}t.
  $$
- **Information Measures and Divergences**:
  - Shannon Entropy (discrete): $H(\mathsf{X}) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$.
  - Differential Entropy (continuous): $h(\mathsf{X}) = -\int_{\mathcal{X}} p(x) \ln p(x) \,\mathrm{d}x$.
  - Kullback-Leibler Divergence:
    $$
      D_{\mathrm{KL}}\bigl(p \,\|\, q\bigr) = \int_{\mathcal{X}} p(x) \ln \frac{p(x)}{q(x)} \,\mathrm{d}x \ge 0.
    $$
  - Fisher Information Matrix:
    $$
      \mathbf{I}(\bm{\theta}) = \mathbb{E}_{\mathsf{X} \sim p(\cdot;\bm{\theta})}\bigl[ \nabla_{\bm{\theta}} \ln p(\mathsf{X}; \bm{\theta}) \, \nabla_{\bm{\theta}} \ln p(\mathsf{X}; \bm{\theta})^\top \bigr] = -\mathbb{E}\bigl[ \nabla_{\bm{\theta}}^2 \ln p(\mathsf{X}; \bm{\theta}) \bigr].
    $$
  - Wasserstein Distance ($L^p$-Optimal Transport):
    $$
      \mathcal{W}_p(\mu, \nu) = \left( \inf_{\gamma \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{X}} d(x, y)^p \,\mathrm{d}\gamma(x, y) \right)^{1/p}.
    $$

---

### F. Function Spaces, Distributions, and Variational Formulations
- **Lebesgue Spaces**:
  $$
    L^p(\Omega) = \left\{ u \colon \Omega \to \mathbb{R} \;\Big|\; \|u\|_{L^p(\Omega)} \equiv \left( \int_\Omega |u(\mathbf{x})|^p \,\mathrm{d}\mathbf{x} \right)^{1/p} < \infty \right\}, \quad \|u\|_{L^\infty(\Omega)} = \operatorname{ess\,sup}_{\mathbf{x} \in \Omega} |u(\mathbf{x})|.
  $$
- **Sobolev Spaces**:
  $$
    W^{k, p}(\Omega) = \bigl\{ u \in L^p(\Omega) \;\big|\; D^\alpha u \in L^p(\Omega) \;\; \forall |\alpha| \le k \bigr\}, \quad H^s(\Omega) \equiv W^{s, 2}(\Omega).
  $$
  Subspaces with vanishing trace on the boundary $\partial\Omega$:
  $$
    H^1_0(\Omega) = \bigl\{ u \in H^1(\Omega) \;\big|\; \gamma_0(u) = 0 \text{ on } \partial\Omega \bigr\}, \quad \text{Trace Space: } H^{1/2}(\partial\Omega).
  $$
- **Vector Sobolev Spaces (Electromagnetics & Fluid Dynamics)**:
  $$
    \mathbf{H}(\operatorname{div}; \Omega) = \bigl\{ \mathbf{v} \in L^2(\Omega)^d \;\big|\; \nabla \cdot \mathbf{v} \in L^2(\Omega) \bigr\}, \quad
    \mathbf{H}(\operatorname{curl}; \Omega) = \bigl\{ \mathbf{v} \in L^2(\Omega)^3 \;\big|\; \nabla \times \mathbf{v} \in L^2(\Omega)^3 \bigr\}.
  $$
- **Duality Pairings and Hilbert Inner Products**:
  - Inner product in $L^2(\Omega)$: $(u, v)_{L^2(\Omega)} = \int_\Omega u(\mathbf{x}) \overline{v(\mathbf{x})} \,\mathrm{d}\mathbf{x}$.
  - Dual pairing: $\langle f, v \rangle_{V^*, V}$ or $\langle T, \phi \rangle_{\mathcal{D}'(\Omega), \mathcal{D}(\Omega)}$.
- **Test Functions and Generalized Distributions**:
  - Smooth functions with compact support: $C^\infty_c(\Omega) \equiv \mathcal{D}(\Omega)$.
  - Schwartz space of rapidly decreasing functions: $\mathcal{S}(\mathbb{R}^d) = \bigl\{ \phi \in C^\infty(\mathbb{R}^d) \;\big|\; \sup_{\mathbf{x}} |\mathbf{x}^\alpha D^\beta \phi(\mathbf{x})| < \infty \bigr\}$.
  - Space of tempered distributions: $\mathcal{S}'(\mathbb{R}^d)$.

---

### G. Quantum States, Operator Algebras, and Spectral Theory
- **Dirac Bra-Ket State Space**:
  $$
    \lvert \psi \rangle \in \mathcal{H}, \quad \langle \phi \rvert \in \mathcal{H}^*, \quad \langle \phi \mid \psi \rangle \in \mathbb{C}, \quad \lvert \psi \rangle \langle \phi \rvert \in \mathcal{B}(\mathcal{H}).
  $$
- **Quantum Operators**: Denote operators by Roman letters with a circumflex ($\hat{H}, \hat{p}, \hat{x}, \hat{a}^\dagger, \hat{a}$).
- **Density Operator (Quantum State)**:
  $$
    \hat{\rho} = \sum_i p_i \lvert \psi_i \rangle \langle \psi_i \rvert, \quad \hat{\rho}^\dagger = \hat{\rho}, \quad \hat{\rho} \ge 0, \quad \operatorname{Tr}(\hat{\rho}) = 1.
  $$
- **Commutators, Anticommutators, and Poisson Brackets**:
  - Quantum Commutator: $[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}$.
  - Anticommutator: $\{\hat{A}, \hat{B}\} = \hat{A}\hat{B} + \hat{B}\hat{A}$.
  - Canonical Commutation Relations: $[\hat{x}_j, \hat{p}_k] = \mathrm{i}\hbar \delta_{jk} \hat{I}, \quad \{\hat{\psi}_\alpha, \hat{\psi}_\beta^\dagger\} = \delta_{\alpha\beta} \hat{I}$.
  - Dirac Classical-Quantum Correspondence: $[\hat{A}, \hat{B}] \longleftrightarrow \mathrm{i}\hbar \{A, B\}_{\mathrm{PB}}$.
- **Spectral Theory of Linear Operators**:
  - Spectrum: $\sigma(A) = \sigma_{\mathrm{p}}(A) \cup \sigma_{\mathrm{c}}(A) \cup \sigma_{\mathrm{r}}(A) \subset \mathbb{C}$ (point, continuous, residual).
  - Resolvent Set & Resolvent Operator: $\rho(A) = \mathbb{C} \setminus \sigma(A), \quad R_z(A) = (A - z I)^{-1} \in \mathcal{B}(\mathcal{H})$.
  - Spectral Decomposition (Self-Adjoint Operator): $A = \int_{\sigma(A)} \lambda \,\mathrm{d}E(\lambda)$.

---

### H. Subscript, Superscript, and Modifier Typography (Disambiguation Hierarchy)
- **Running Index Variables (Italic Math)**:
  $a_i, x_k, v_j, \sum_{n=1}^N$ (indices $i, j, k, n \in \mathbb{N}$ represent mathematical running variables).
- **Descriptive Textual Labels (Upright Roman)**:
  $v_{\mathrm{P}}$ ($P$-wave velocity), $v_{\mathrm{S}}$ ($S$-wave velocity), $E_{\mathrm{kin}}$ (kinetic energy), $E_{\mathrm{pot}}$ (potential energy), $T_{\mathrm{eff}}$ (effective temperature), $\lambda_{\max}$ (maximum eigenvalue), $\omega_{\mathrm{crit}}$ (critical frequency), $t_{\mathrm{arr}}$ (arrival time), $\mathbf{f}_{\mathrm{ext}}$ (external force).
- **Operator Modifiers & Transpositions**:
  - Transpose: $\mathbf{A}^\top$ (`\mathbf{A}^\top`) or $\mathbf{A}^{\mathsf{T}}$ (`\mathbf{A}^{\mathsf{T}}`), NEVER bare italic $T$ (`\mathbf{A}^T`, which collides with temperature or period $T$).
  - Hermitian Conjugate (Adjoint): $\mathbf{A}^\dagger$ (`\mathbf{A}^\dagger`) or $A^*$.
  - Complex Conjugate: $\bar{z}$ or $z^*$.
  - Moore-Penrose Pseudoinverse: $\mathbf{A}^+$ or $\mathbf{A}^\dagger$.
  - Set Complement: $A^{\mathrm{c}}$ (`A^{\mathrm{c}}`).
  - Orthogonal Complement: $V^\perp$ (`V^\perp`).

---

### J. Algorithmic Complexity, Asymptotics, and Computational Problem Classes
In theoretical computer science and algorithm analysis, express asymptotic computational bounds, complexity classes, and formal decision problems with invariant typographic discipline:

- **Bachmann-Landau Asymptotic Notations**:
  - Upper Bound (Big-O): $f(n) = \mathcal{O}(g(n)) \iff \exists c > 0, n_0 \in \mathbb{N} : \forall n \ge n_0, \; |f(n)| \le c |g(n)|$.
  - Lower Bound (Big-Omega): $f(n) = \Omega(g(n)) \iff \exists c > 0, n_0 \in \mathbb{N} : \forall n \ge n_0, \; f(n) \ge c g(n) \ge 0$.
  - Tight Bound (Big-Theta): $f(n) = \Theta(g(n)) \iff f(n) = \mathcal{O}(g(n)) \text{ and } f(n) = \Omega(g(n))$.
  - Strict Upper Bound (Little-o): $f(n) = o(g(n)) \iff \lim_{n \to \infty} \frac{f(n)}{g(n)} = 0$.
  - Strict Lower Bound (Little-omega): $f(n) = \omega(g(n)) \iff \lim_{n \to \infty} \frac{f(n)}{g(n)} = \infty$.
  - Soft-O (Polylogarithmic Suppression): $\widetilde{\mathcal{O}}(g(n)) \equiv \mathcal{O}\bigl(g(n) \log^k g(n)\bigr)$ for some $k \ge 0$.
- **Complexity Classes (Upright Sans-Serif)**:
  - Deterministic & Nondeterministic Polynomial Time: $\mathsf{P}, \mathsf{NP}, \mathsf{co\text{-}NP}, \mathsf{PSPACE}, \mathsf{EXPTIME}, \mathsf{L}, \mathsf{NL}$.
  - Quantum & Randomized: $\mathsf{BQP}, \mathsf{BPP}, \mathsf{RP}, \mathsf{ZPP}$.
  - Parallel Circuits: $\mathsf{NC}^k, \mathsf{AC}^k, \mathsf{TC}^k$.
  - Polynomial Hierarchy: $\Sigma_k^{\mathsf{P}}, \Pi_k^{\mathsf{P}}, \Delta_k^{\mathsf{P}}$, with total hierarchy $\mathsf{PH} = \bigcup_{k \ge 0} \Sigma_k^{\mathsf{P}}$.
  - Counting & Hardness Classes: $\#\mathsf{P}$ (`\#\mathsf{P}`), $\mathsf{NP}\text{-hard}, \mathsf{NP}\text{-complete}$.
- **Canonical Computational Problems (Small Caps)**:
  - Decision & Optimization Problems: $\textsc{3-Sat}, \textsc{MaxCut}, \textsc{VertexCover}, \textsc{SubsetSum}, \textsc{ShortestPath}, \textsc{TravellingSalesperson}$.
- **Formal Reductions**:
  - Polynomial-Time Many-One (Karp): $A \le_{\mathrm{P}} B$ (or $A \le_{\mathrm{m}}^{\mathrm{P}} B$).
  - Polynomial-Time Turing (Cook): $A \le_{\mathrm{T}}^{\mathrm{P}} B$.
  - Logarithmic Space: $A \le_{\log} B$.
- **Formal Automata & Turing Machines**:
  - Deterministic Finite Automaton (DFA): 5-tuple $\mathcal{M} = (Q, \Sigma, \delta, q_0, F)$ with state set $Q$, input alphabet $\Sigma$, transition function $\delta \colon Q \times \Sigma \to Q$, initial state $q_0 \in Q$, and accepting states $F \subseteq Q$.
  - Pushdown Automaton (PDA): 6-tuple $\mathcal{P} = (Q, \Sigma, \Gamma, \delta, q_0, F)$ with stack alphabet $\Gamma$.
  - Turing Machine (TM): 7-tuple $\mathcal{T} = (Q, \Sigma, \Gamma, \delta, q_0, q_{\mathrm{accept}}, q_{\mathrm{reject}})$ where $\delta \colon Q \times \Gamma \to Q \times \Gamma \times \{\mathrm{L}, \mathrm{R}\}$.
  - String & Language Operations: Empty string $\varepsilon$, language $L \subseteq \Sigma^*$, Kleene star closure $\Sigma^*$, positive closure $\Sigma^+$.

---

### K. Machine Learning, Deep Neural Architectures, and Statistical Learning Theory
Typeset neural networks, multidimensional data tensors, transformer attention operators, and generalization bounds with explicit dimension signatures:

- **Multidimensional Data Tensors**:
  - High-Order Tensors: Bold calligraphic or Euler script ($\bm{\mathcal{X}} \in \mathbb{R}^{B \times C \times H \times W}$ for Batch $\times$ Channels $\times$ Height $\times$ Width).
  - Mode-$n$ Matricization (Unfolding): $\mathbf{X}_{(n)} \in \mathbb{R}^{I_n \times \prod_{j \ne n} I_j}$.
  - Mode-$n$ Tensor-Matrix Contraction: $\bm{\mathcal{Y}} = \bm{\mathcal{X}} \times_n \mathbf{U} \in \mathbb{R}^{I_1 \times \dots \times J_n \times \dots \times I_N}$.
- **Neural Layer Architecture & Parameters**:
  - Layer Indices: Parenthesized superscript for depth:
    $$
      \mathbf{h}^{(l)} = \sigma\left( \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)} \right), \quad \mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}, \quad \mathbf{b}^{(l)} \in \mathbb{R}^{d_l}.
    $$
  - Pre-activation vs. Activation: $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \quad \mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$.
  - Latent Bottlenecks & Embeddings: Latent vector $\mathbf{z} \sim q_{\bm{\phi}}(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\bm{\mu}_{\bm{\phi}}(\mathbf{x}), \operatorname{diag}(\bm{\sigma}_{\bm{\phi}}^2(\mathbf{x})))$.
- **Transformer Algebra & Scaled Attention Mechanics**:
  - Projection Weights: Query $\mathbf{W}_Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$, Key $\mathbf{W}_K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$, Value $\mathbf{W}_V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}$.
  - Sequence Embeddings: Sequence matrix $\mathbf{X} \in \mathbb{R}^{N \times d_{\mathrm{model}}}$, queries $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, keys $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, values $\mathbf{V} = \mathbf{X}\mathbf{W}_V$.
  - Scaled Dot-Product Attention:
    $$
      \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}, \quad \mathbf{M} \in \{0, -\infty\}^{N \times M}.
    $$
  - Multi-Head Attention:
    $$
      \operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Concat}(\operatorname{head}_1, \dots, \operatorname{head}_h) \mathbf{W}_O, \quad \operatorname{head}_i = \operatorname{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V).
    $$
- **Empirical Risk Minimization (ERM) & Optimization**:
  - Empirical Loss & Regularization:
    $$
      \widehat{\mathcal{R}}_n(\bm{\theta}) = \frac{1}{n}\sum_{i=1}^n \ell\bigl(f_{\bm{\theta}}(\mathbf{x}_i), y_i\bigr) + \lambda \Omega(\bm{\theta}), \quad \Omega(\bm{\theta}) \in \left\{ \|\bm{\theta}\|_1, \, \frac{1}{2}\|\bm{\theta}\|_2^2 \right\}.
    $$
  - Population Generalization Risk: $\mathcal{R}(\bm{\theta}) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}\bigl[ \ell(f_{\bm{\theta}}(\mathbf{x}), y) \bigr]$.
  - Adaptive Stochastic Gradient Updates (AdamW):
    $$
      \mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^{\odot 2}, \quad \bm{\theta}_t = \bm{\theta}_{t-1} - \eta_t \left( \frac{\widehat{\mathbf{m}}_t}{\sqrt{\widehat{\mathbf{v}}_t} + \epsilon} + \gamma \bm{\theta}_{t-1} \right).
    $$
- **Statistical Learning Theory & Generalization Bounds**:
  - Empirical Rademacher Complexity:
    $$
      \widehat{\mathfrak{R}}_S(\mathcal{H}) = \mathbb{E}_{\bm{\sigma} \in \{-1, +1\}^n}\left[ \sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \sigma_i h(\mathbf{x}_i) \right].
    $$
  - Vapnik-Chervonenkis (VC) Dimension: $\operatorname{VCdim}(\mathcal{H}) \in \mathbb{N}$.
  - PAC Generalization Bound: For any $\delta \in (0, 1)$, with probability at least $1 - \delta$ over sample $S \sim \mathcal{D}^n$:
    $$
      \mathcal{R}(h) \le \widehat{\mathcal{R}}_S(h) + 2\mathfrak{R}_n(\mathcal{H}) + \sqrt{\frac{\ln(2/\delta)}{2n}}.
    $$

---

### L. Spectral Graph Theory, Discrete Networks, and Graph Neural Networks
Format discrete topological relations, graph Laplacians, and spatial message-passing architectures consistently:

- **Graph Structure**: Discrete graph $\mathsf{G} = (\mathsf{V}, \mathsf{E}, \mathbf{W})$ with vertex set $\mathsf{V} = \{v_1, \dots, v_n\}$ ($|\mathsf{V}| = n$), edge set $\mathsf{E} \subseteq \mathsf{V} \times \mathsf{V}$ ($|\mathsf{E}| = m$), and optional edge weight matrix $\mathbf{W} \in \mathbb{R}_{\ge 0}^{n \times n}$.
- **Adjacency and Degree Matrices**:
  - Unweighted Adjacency: $\mathbf{A} \in \{0, 1\}^{n \times n}$ where $A_{uv} = 1 \iff (u, v) \in \mathsf{E}$.
  - Degree Diagonal Matrix: $\mathbf{D} = \operatorname{diag}(d_1, \dots, d_n)$ where $d_u = \sum_{v \in \mathsf{V}} A_{uv}$.
- **Graph Laplacians & Quadratic Forms**:
  - Combinatorial (Unnormalized) Laplacian: $\mathbf{L} = \mathbf{D} - \mathbf{A}$.
  - Symmetric Normalized Laplacian: $\mathbf{L}_{\mathrm{sym}} = \mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2} = \mathbf{I}_n - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$.
  - Random-Walk Normalized Laplacian: $\mathbf{L}_{\mathrm{rw}} = \mathbf{D}^{-1}\mathbf{L} = \mathbf{I}_n - \mathbf{D}^{-1}\mathbf{A}$.
  - Dirichlet Quadratic Energy:
    $$
      \mathbf{x}^\top \mathbf{L} \mathbf{x} = \frac{1}{2}\sum_{(u, v) \in \mathsf{E}} W_{uv}(x_u - x_v)^2 \ge 0 \quad \forall \mathbf{x} \in \mathbb{R}^n.
    $$
  - Spectral Decomposition: $\mathbf{L} = \mathbf{U} \bm{\Lambda} \mathbf{U}^\top$ with orthonormal eigenvectors $\mathbf{U} = [\mathbf{u}_1, \dots, \mathbf{u}_n]$ and eigenvalues $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$.
- **Graph Neural Network (GNN) Message Passing**:
  - Spatial Vertex Aggregation & Update:
    $$
      \mathbf{h}_v^{(k)} = \operatorname{UPDATE}^{(k)}\left( \mathbf{h}_v^{(k-1)}, \, \operatorname{AGGREGATE}^{(k)}\left( \bigl\{ \mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v) \bigr\} \right) \right), \quad \mathcal{N}(v) = \{u \in \mathsf{V} : (u, v) \in \mathsf{E}\}.
    $$
  - Spectral Graph Convolution (GCN 1st-Order Chebyshev Approximation):
    $$
      \mathbf{H}^{(k)} = \sigma\left( \widetilde{\mathbf{D}}^{-1/2} \widetilde{\mathbf{A}} \widetilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(k-1)} \mathbf{W}^{(k)} \right), \quad \widetilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_n, \quad \widetilde{\mathbf{D}}_{ii} = \sum_j \widetilde{A}_{ij}.
    $$

---

### M. Formal Methods, Type Theory, and Program Semantics
In programming languages, formal semantics, and software verification, apply unambiguous typeface discipline to judgments, reduction relations, and type hierarchies:

- **Typing Judgments & Proof Rules**:
  - Typing Derivation: $\Gamma \vdash e : \tau$ (in typing context $\Gamma$, expression $e$ has type $\tau$).
  - Function Abstraction & Application Rules:
    $$
      \frac{\Gamma, x \colon \tau_1 \vdash e : \tau_2}{\Gamma \vdash (\lambda x \colon \tau_1.\, e) : \tau_1 \to \tau_2}, \quad
      \frac{\Gamma \vdash e_1 : \tau_1 \to \tau_2 \quad \Gamma \vdash e_2 : \tau_1}{\Gamma \vdash e_1 \, e_2 : \tau_2}.
    $$
- **Operational Semantics**:
  - Small-Step Reduction: $e \to e'$ or state transition $\langle c, \sigma \rangle \to \langle c', \sigma' \rangle$.
  - Reflexive-Transitive Multi-Step Closure: $e \to^* e'$.
  - Big-Step Natural Evaluation: $\langle e, \sigma \rangle \Downarrow \langle v, \sigma' \rangle$.
- **Denotational Semantics**:
  - Semantic Interpretation Brackets: $\llbracket e \rrbracket_\rho \in \mathcal{D}$ maps syntax to a mathematical semantic domain $\mathcal{D}$ under environment $\rho$.
- **Type Hierarchy & Constructors**:
  - Base Primitive Types (Sans-Serif): $\mathsf{Nat}, \mathsf{Bool}, \mathsf{Unit}, \mathsf{String}, \mathsf{Float32}$.
  - Function & Product Types: $\tau_1 \to \tau_2, \quad \tau_1 \times \tau_2, \quad \tau_1 + \tau_2$.
  - Polymorphic Quantifiers: Universal type $\forall X.\, \tau$, Existential type $\exists X.\, \tau$.
  - Dependent Types: Dependent product (Pi-type) $\Pi(x \colon A).\, B(x)$, Dependent sum (Sigma-type) $\Sigma(x \colon A).\, B(x)$.
- **Program Verification & Hoare Logic Triples**:
  - Partial & Total Correctness Triples: $\{P\} \; C \; \{Q\}$ (precondition $P$, command $C$, postcondition $Q$).
  - While Loop Invariant Rule:
    $$
      \frac{\{P \land b\} \; C \; \{P\}}{\{P\} \; \mathbf{while} \; b \; \mathbf{do} \; C \; \{P \land \neg b\}}.
    $$

---

### N. Scientific Computing, Numerical Linear Algebra, and High-Performance Computing (HPC)
Rigorous computational science demands precise specification of machine precision, matrix condition numbers, sparse storage invariants, and parallel scaling metrics:

- **Floating-Point Arithmetic & Error Bounds**:
  - IEEE Precision Formats (Upright Roman): $\mathrm{FP64}$ (double precision), $\mathrm{FP32}$ (single precision), $\mathrm{TF32}$, $\mathrm{BF16}$ (bfloat16), $\mathrm{FP16}$, $\mathrm{FP8}$.
  - Machine Epsilon ($\varepsilon_{\mathrm{mach}}$) & Unit Roundoff ($u$):
    $$
      \operatorname{fl}(x) = x(1 + \delta), \quad |\delta| \le u = \frac{1}{2}\varepsilon_{\mathrm{mach}}, \quad u_{\mathrm{FP64}} = 2^{-53} \approx 1.11 \times 10^{-16}, \quad u_{\mathrm{FP32}} = 2^{-24} \approx 5.96 \times 10^{-8}.
    $$
- **Matrix Condition Numbers & Numerical Stability**:
  - Condition Number with Respect to Inversion:
    $$
      \kappa_p(\mathbf{A}) = \|\mathbf{A}\|_p \|\mathbf{A}^{-1}\|_p, \quad \kappa_2(\mathbf{A}) = \frac{\sigma_{\max}(\mathbf{A})}{\sigma_{\min}(\mathbf{A})}.
    $$
  - Forward vs. Backward Error in Linear Systems $\mathbf{A}\mathbf{x} = \mathbf{b}$:
    $$
      \frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} \le \kappa(\mathbf{A}) \frac{\|\mathbf{r}\|}{\|\mathbf{b}\|}, \quad \text{where residual } \mathbf{r} = \mathbf{b} - \mathbf{A}\hat{\mathbf{x}}.
    $$
- **Krylov Subspace Projections**:
  - $m$-Dimensional Krylov Space:
    $$
      \mathcal{K}_m(\mathbf{A}, \mathbf{r}_0) = \operatorname{span}\bigl\{ \mathbf{r}_0, \mathbf{A}\mathbf{r}_0, \mathbf{A}^2\mathbf{r}_0, \dots, \mathbf{A}^{m-1}\mathbf{r}_0 \bigr\} \subset \mathbb{R}^n.
    $$
- **Sparse Matrix Representations**:
  - Upright Storage Abbreviations:
    - $\mathrm{CSR}$ (Compressed Sparse Row: \texttt{values}, \texttt{col\_indices}, \texttt{row\_ptr}).
    - $\mathrm{CSC}$ (Compressed Sparse Column: \texttt{values}, \texttt{row\_indices}, \texttt{col\_ptr}).
    - $\mathrm{COO}$ (Coordinate List: \texttt{values}, \texttt{row\_indices}, \texttt{col\_indices}).
- **HPC Parallel Scaling and Speedup Laws**:
  - Parallel Speedup: $S_p = \frac{T_1}{T_p}$, Parallel Efficiency: $E_p = \frac{S_p}{p} = \frac{T_1}{p T_p}$.
  - Amdahl's Law (Strong Scaling with strictly serial fraction $s \in [0, 1]$):
    $$
      S_p = \frac{1}{s + \frac{1-s}{p}} \le \frac{1}{s}.
    $$
  - Gustafson-Barsis' Law (Weak Scaling with scaled workload and serial fraction $\alpha$):
    $$
      S_p = p - \alpha(p - 1).
    $$
  - Roofline Model & Arithmetic Intensity:
    $$
      I = \frac{\text{FLOPs}}{\text{DRAM Traffic (Bytes)}} \quad \left[\unit{\text{FLOP}\per\byte}\right], \quad
      P_{\max} = \min\left( P_{\mathrm{peak}}, \, I \times B_{\mathrm{peak}} \right).
    $$

---

### O. Cryptography, Bilinear Pairings, and Information Security
In theoretical cryptography and cybersecurity, adhere to strict asymptotic security parameter conventions:

- **Security Parameters & Keys**:
  - Security parameter in unary: $1^\lambda$ where $\lambda \in \mathbb{N}$.
  - Public / Secret Key Pairs: $(pk, sk) \in \mathcal{PK} \times \mathcal{SK}$ or $(vk, sk)$ for verification/signing.
  - Core Cryptographic Algorithms (Sans-Serif):
    $$
      \mathsf{KeyGen}(1^\lambda) \to (pk, sk), \quad \mathsf{Enc}_{pk}(m) \to c, \quad \mathsf{Dec}_{sk}(c) \to m.
    $$
    Digital Signatures: $\mathsf{Sign}_{sk}(m) \to \sigma, \quad \mathsf{Verify}_{pk}(m, \sigma) \to \{0, 1\}$.
- **Negligible Functions**:
  $$
    \mu(\lambda) \in \operatorname{negl}(\lambda) \iff \forall c \in \mathbb{N}, \; \exists \lambda_0 \in \mathbb{N} : \forall \lambda \ge \lambda_0, \; \mu(\lambda) < \lambda^{-c}.
  $$
- **Adversarial Advantage & Security Games**:
  $$
    \operatorname{\mathbf{Adv}}_{\mathcal{A}}^{\mathrm{IND\text{-}CPA}}(\lambda) = \left| \mathbb{P}\left[ \mathsf{Exp}_{\mathcal{A}}^{\mathrm{IND\text{-}CPA}\text{-}1}(\lambda) = 1 \right] - \mathbb{P}\left[ \mathsf{Exp}_{\mathcal{A}}^{\mathrm{IND\text{-}CPA}\text{-}0}(\lambda) = 1 \right] \right| \le \operatorname{negl}(\lambda).
  $$
- **Bilinear Pairings on Elliptic Curves**:
  - Pairing Map: Non-degenerate, bilinear map $e \colon \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ between cyclic groups of prime order $p$:
    $$
      e(aP, bQ) = e(P, Q)^{ab} \quad \forall P \in \mathbb{G}_1, \; Q \in \mathbb{G}_2, \; a, b \in \mathbb{Z}_p.
    $$
- **Random Oracles & Cryptographic Hash Functions**:
  $$
    \mathsf{H} \colon \{0, 1\}^* \to \{0, 1\}^n, \quad \mathcal{O}_{\mathrm{RO}} \colon \mathcal{M} \to \mathcal{C}.
  $$

---

### P. Semantic Predefined Operators (`\DeclareMathOperator`)
Declare mathematical, physical, and computational operators in the LaTeX preamble using `\DeclareMathOperator` (or `\DeclareMathOperator*` for operators with subscript limits):
```latex
% Linear Algebra & Matrix Analysis
\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\nullity}{null}
\DeclareMathOperator{\spanop}{span}
\DeclareMathOperator{\cond}{cond}

% Differential & Vector Calculus
\DeclareMathOperator{\grad}{grad}
\DeclareMathOperator{\divop}{div}
\DeclareMathOperator{\curl}{curl}
\DeclareMathOperator{\rot}{rot}
\DeclareMathOperator{\supp}{supp}
\DeclareMathOperator{\diam}{diam}
\DeclareMathOperator{\vol}{vol}

% Optimization & Variational Analysis (with limits)
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\esssup}{ess\,sup}
\DeclareMathOperator*{\essinf}{ess\,inf}

% Probability, Statistics & Information Theory
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Cov}{Cov}
\DeclareMathOperator{\Corr}{Corr}
\DeclareMathOperator{\Bias}{Bias}
\DeclareMathOperator{\MSE}{MSE}

% Machine Learning & Neural Network Operators
\DeclareMathOperator{\softmax}{softmax}
\DeclareMathOperator{\ReLU}{ReLU}
\DeclareMathOperator{\GELU}{GELU}
\DeclareMathOperator{\SiLU}{SiLU}
\DeclareMathOperator{\Attention}{Attention}
\DeclareMathOperator{\MultiHead}{MultiHead}
\DeclareMathOperator{\Concat}{Concat}
\DeclareMathOperator{\vecop}{vec}

% Computational Complexity & Learning Theory
\DeclareMathOperator{\poly}{poly}
\DeclareMathOperator{\polylog}{polylog}
\DeclareMathOperator{\negl}{negl}
\DeclareMathOperator{\VCdim}{VCdim}

% Graph Theory & Discrete Networks
\DeclareMathOperator{\degop}{deg}
\DeclareMathOperator{\volG}{vol}
\DeclareMathOperator{\Cut}{Cut}

% Scientific Computing & Numerical Arithmetic
\DeclareMathOperator{\fl}{fl}
\DeclareMathOperator{\nnz}{nnz}

% Cryptography & Security Games
\DeclareMathOperator{\Adv}{\mathbf{Adv}}
\DeclareMathOperator{\Exp}{\mathbf{Exp}}

% Complex Analysis & Signal Processing
\DeclareMathOperator{\Reop}{Re}
\DeclareMathOperator{\Imop}{Im}
\DeclareMathOperator{\res}{res}
\DeclareMathOperator{\sgn}{sgn}
\DeclareMathOperator{\sinc}{sinc}
\DeclareMathOperator{\erf}{erf}
```

---

## 3. Step-by-Step Derivation Archetype and Epistemic Relational Operators

### A. The "Deductive-Terminal" Relational Synthesis and Attention Conditioning

Autoregressive language models have an overwhelming pre-trained bias toward standard equality (`=`, `&=`) following a mathematical expression. Demanding non-standard relational symbols like `\stackrel{?}{=}` (conjectured equality) or `\stackrel{!}{=}` (demanded equality) in an unconditioned context forces attention heads to fight base weights, causing perplexity spikes, syntactic corruption (`\overset{?}=`, `& = ?`), or regression to `=`.

To eliminate epistemic conflation while maintaining mathematical and typographic elegance, adhere to the **Deductive-Terminal Relational Principle**:
1. **Strict Non-Conflation**: Bare equality (`=`) is **strictly forbidden** for definitions, global identities, variational demands, stationarity constraints, and unverified conjectures.
2. **Deductive-Terminal Role**: Bare `=` is **retained exclusively** for propositional equality resulting from direct deductive algebraic evaluation, substitution, or simplification steps following inescapably from established equations.
3. **Prefix Conditioning**: To align with transformer attention mechanics, always prime the autoregressive context by stating the epistemic intent in the preceding prose or leading annotation *before* emitting the relational token.

| Relational Category | Canonical Syntax | Mathematical Role & Meaning | Recommended Usage | Style Preference & Notes | Explicit Intent / Contextual Anchor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Definition** | `\coloneqq` (or `\triangleq`) | Binds a newly introduced symbol or functional to an expression | Functional introductions, parameter definitions | Never use bare `=` | *"We define...", "Setting...", "By convention..."* |
| **Global Identity** | `\equiv` | Invariant holding identically across the entire domain, manifold, or function space | Geometric/algebraic invariants, structural identities | Never use bare `=` | *"By tensor symmetry...", "Identically vanishing over $\Omega$..."* |
| **Variational Demand** | `\stackrel{!}{=}` or `&\stackrel{!}{=}` (or `\eqreq`) | Optimality condition, stationarity constraint, Hamilton principle | Euler-Lagrange extremum, stationarity requirements | Never use bare `=` | *"Demanding stationarity...", "Requiring vanishing boundary flux..."* |
| **Candidate / Conjecture** | `\stackrel{?}{=}` or `&\stackrel{?}{=}` (or `\eqquest`) | Unverified candidate, trial ansatz check, identity to prove | Hypothesis checks, trial solution substitutions | Never use bare `=` | *"To verify whether...", "Testing candidate solution..."* |
| **Deductive-Terminal** | `=` or `&=` | Propositional equality resulting from direct algebraic evaluation or substitution | Term expansion, algebraic reduction, evaluating an identity | Never use for definitions, demands, or conjectures | *"Direct substitution yields...", "Collecting terms...", "Evaluating..."* |
| **Asymptotic / Scaling** | `\sim`, `\simeq`, `\approx`, `\asymp` | Asymptotic scaling limit, diffeomorphism, or numerical estimate | High/low frequency limits, far-field approximations | Never use bare `=` | *"In the high-frequency eikonal limit...", "Scaling asymptotically as..."* |

> [!TIP]
> **Token Chunking and Macro Discipline**:
> In multi-line derivations, place the ampersand directly before the relation (`&\stackrel{?}{=}`, `&\coloneqq`, `&\stackrel{!}{=}`, `&=`) without rogue spaces. For high-frequency use in preambles, declare semantic relation macros to prevent token fragmentation across braces:
> ```latex
> \newcommand{\defeq}{\coloneqq}                   % Definitional equality (requires mathtools)
> \newcommand{\eqident}{\equiv}                    % Global identity across domain
> \newcommand{\eqreq}{\mathrel{\stackrel{!}{=}}}   % Demanded / variational equality
> \newcommand{\eqquest}{\mathrel{\stackrel{?}{=}}} % Conjectured / candidate equality
> ```

### B. Canonical Derivation Archetype with Epistemic State Transitions

When presenting multi-step algebraic or physical derivations, align relation symbols according to their precise epistemic category and place natural-language justifications in a right-anchored column using `&& \text{...}`:

```latex
\begin{align}
  % Step 1: Definition of the action functional (Definitional equality: \coloneqq)
  \mathcal{S}[\mathbf{u}]
    &\coloneqq \int_{t_0}^{t_1} \int_\Omega \mathcal{L}(\mathbf{u}, \nabla\mathbf{u}, \dot{\mathbf{u}}) \,\mathrm{d}\Omega \,\mathrm{d}t
    && \text{Action functional definition} \label{eq:action_def} \\
  % Step 2: Stationarity demand (Variational demand: \stackrel{!}{=}, never bare =)
  \delta \mathcal{S}[\mathbf{u}]
    &\stackrel{!}{=} 0
    && \text{Stationarity demand: Hamilton's variational principle} \label{eq:stationarity_demand} \\
  % Step 3: Implication to Euler-Lagrange equations of motion
  \implies \frac{\partial \mathcal{L}}{\partial \mathbf{u}} - \nabla \cdot \frac{\partial \mathcal{L}}{\partial (\nabla\mathbf{u})} - \frac{\partial}{\partial t}\frac{\partial \mathcal{L}}{\partial \dot{\mathbf{u}}}
    &= \mathbf{0}
    && \text{Euler--Lagrange equation (deductive consequence)} \label{eq:euler_lagrange} \\
  % Step 4: Algebraic substitution of linear elastodynamic Lagrangian density
  &= \rho \ddot{\mathbf{u}} - \nabla \cdot \bm{\sigma} - \mathbf{f}
    && \text{Direct substitution of linear elastic density} \label{eq:momentum_balance} \\
  % Step 5: Constitutive closure definition (\coloneqq)
  \bm{\sigma}
    &\coloneqq \lambda (\nabla \cdot \mathbf{u}) \mathbf{I} + \mu \bigl( \nabla \mathbf{u} + (\nabla \mathbf{u})^\top \bigr)
    && \text{Isotropic linear constitutive closure} \label{eq:isotropic_hooke}
\end{align}
Substituting the constitutive closure~\eqref{eq:isotropic_hooke} into momentum balance~\eqref{eq:momentum_balance} with homogeneous Lam\'e parameters ($\nabla\lambda = \nabla\mu = \mathbf{0}$) yields the deductive evaluation:
$$
  (\lambda + \mu) \nabla (\nabla \cdot \mathbf{u}) + \mu \nabla^2 \mathbf{u} + \mathbf{f} = \rho \ddot{\mathbf{u}}.
$$
Taking the divergence of both sides and defining $\theta \coloneqq \nabla \cdot \mathbf{u}$ with zero body force ($\mathbf{f} = \mathbf{0}$) reduces the compressional motion to the scalar Helmholtz wave equation:
\begin{equation}
  \nabla^2 \theta - \frac{1}{\alpha^2} \frac{\partial^2 \theta}{\partial t^2} = 0, \quad \text{where} \quad \alpha \coloneqq \sqrt{\frac{\lambda + 2\mu}{\rho}}.
  \label{eq:p_wave_equation}
\end{equation}
```

For candidate verification, anchor the hypothesis check before the relation:
```latex
% Candidate verification check: hypothesis test before \stackrel{?}{=}
To test whether a proposed trial wavefield \mathbf{u}_{\mathrm{trial}}(\mathbf{x}, t) satisfies the elastodynamic operator, we evaluate:
\begin{equation}
  \mathcal{L}_{\mathrm{wave}}[\mathbf{u}_{\mathrm{trial}}] \stackrel{?}{=} \mathbf{0}, \quad \text{subject to } \mathbf{u}|_{\Gamma_D} = \mathbf{g}.
  \label{eq:candidate_check}
\end{equation}
```

---

## 4. Formal Theorem-Proof Architecture

Format formal mathematical assertions with `amsthm` environments linked to `cleveref`. Conclude proofs cleanly using `\qedhere` inside display equations:

```latex
\begin{theorem}[Spectral Convergence Rate]
\label{thm:spectral_convergence}
Let $\mathcal{H}$ be a separable Hilbert space, and let $A \colon \mathcal{D}(A) \to \mathcal{H}$ be a linear, self-adjoint, strictly positive operator with compact resolvent. Under the Galerkin projection onto the $N$-dimensional subspace $V_N = \operatorname{span}\{\phi_1, \dots, \phi_N\}$, the finite-element error satisfies:
\begin{equation}
  \|u - u_N\|_{\mathcal{H}} \le C \lambda_{N+1}^{-1/2} \|f\|_{\mathcal{H}},
  \label{eq:error_bound}
\end{equation}
where $\lambda_{N+1}$ is the $(N+1)$-th eigenvalue of $A$, and $C > 0$ is an invariant constant depending only on $\Omega$.
\end{theorem}

\begin{proof}
By the spectral decomposition theorem, the orthonormal eigenfunctions $\{\phi_k\}_{k=1}^\infty$ form a complete basis for $\mathcal{H}$. Expanding the solution $u = \sum_{k=1}^\infty c_k \phi_k$ and subtracting the orthogonal projection $u_N = \sum_{k=1}^N c_k \phi_k$ yields the truncation residual:
\begin{align*}
  \|u - u_N\|_{\mathcal{H}}^2
    &= \sum_{k=N+1}^\infty |c_k|^2 \\
    &\le \frac{1}{\lambda_{N+1}} \sum_{k=N+1}^\infty \lambda_k |c_k|^2 \\
    &\le \frac{1}{\lambda_{N+1}} \|u\|_{A}^2. \qedhere
\end{align*}
\end{proof}
```

---

## 5. Physical Quantities, Dimensions, and SI Units (`siunitx` v3)

In scientific and physical manuscripts, every measurable quantity $Q$ must adhere to the fundamental metrological relation $Q = \{Q\} \cdot [Q]$, expressing the product of a pure numerical value $\{Q\}$ and an invariant physical unit $[Q]$ (BIPM SI Brochure, 9th Edition). Typeset all physical quantities with `siunitx` v3 to enforce invariant upright roman unit fonts, non-breaking thin spaces, and ISO 80000-1 dimensional rigor.

---

### A. The 7 SI Base Units and Defining Invariant Constants
The International System of Units (SI) is founded on seven base physical dimensions, anchored by seven exact defining fundamental physical constants (2019 redefinition):

| Base Dimension | Dimensional Symbol | Base SI Unit | Unit Symbol | Defining Physical Constant | Defining Constant Value & Invariant Anchor |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **Time** | $\mathsf{T}$ | second | $\unit{\second}$ | Hyperfine transition of $^{133}\mathrm{Cs}$ ($\Delta \nu_{\mathrm{Cs}}$) | $\Delta \nu_{\mathrm{Cs}} = \qty{9192631770}{\hertz}$ |
| **Length** | $\mathsf{L}$ | meter | $\unit{\meter}$ | Speed of light in vacuum ($c$) | $c = \qty{299792458}{\meter\per\second}$ |
| **Mass** | $\mathsf{M}$ | kilogram | $\unit{\kilo\gram}$ | Planck constant ($h$) | $h = \qty{6.62607015e-34}{\joule\second}$ |
| **Electric Current** | $\mathsf{I}$ | ampere | $\unit{\ampere}$ | Elementary charge ($e$) | $e = \qty{1.602176634e-19}{\coulomb}$ |
| **Thermodynamic Temperature** | $\mathsf{\Theta}$ | kelvin | $\unit{\kelvin}$ | Boltzmann constant ($k_{\mathrm{B}}$) | $k_{\mathrm{B}} = \qty{1.380649e-23}{\joule\per\kelvin}$ |
| **Amount of Substance** | $\mathsf{N}$ | mole | $\unit{\mole}$ | Avogadro constant ($N_{\mathrm{A}}$) | $N_{\mathrm{A}} = \qty{6.02214076e23}{\per\mole}$ |
| **Luminous Intensity** | $\mathsf{J}$ | candela | $\unit{\candela}$ | Luminous efficacy ($K_{\mathrm{cd}}$ at $\qty{540}{\tera\hertz}$) | $K_{\mathrm{cd}} = \qty{683}{\lumen\per\watt}$ |

---

### B. Dedicated `siunitx` v3 Macro Roles
- `\qty{value}{unit}`: Formats a numerical quantity combined with an SI unit (e.g., `\qty{15}{\kilo\meter}`).
- `\unit{unit}`: Formats standalone units in text, axis labels, or table headers (e.g., `[\unit{\kilo\meter\per\second}]`).
- `\num{value}`: Formats dimensionless scientific numbers, groupings, and floating-point data (e.g., `\num{6.022e23}`, `\num{-4.2}`).
- `\qtyrange{start}{stop}{unit}`: Formats closed numerical intervals with units (e.g., `\qtyrange{5}{15}{\mega\pascal}`).
- `\qtylist{val1; val2; val3}{unit}`: Formats discrete lists of quantities sharing a unit (e.g., `\qtylist{1; 2; 5; 10}{\hertz}`).
- `\ang{deg;min;sec}`: Formats planar angles, geographical coordinates, and inclinations (e.g., `\ang{45;30;12}`, `\ang{12.4}`).
- `\complexqty{a + bi}{unit}`: Formats complex electrical impedance or frequency response (e.g., `\complexqty{50 + 20i}{\ohm}`).

---

### C. Comprehensive Domain-by-Domain Unit Translation Tables

#### 1. Classical Mechanics, Kinematics & Gravitation
| Physical Quantity | Dimensional Formula | Canonical `siunitx` Macro | Rendered Output | Contextual Notes & Equivalences |
| :--- | :--- | :--- | :--- | :--- |
| **Displacement / Length** | $\mathsf{L}$ | `\qty{10}{\kilo\meter}`, `\qty{250}{\micro\meter}`, `\qty{1.5}{\angstrom}` | $10\text{ km}$, $250\text{ }\mu\text{m}$, $1.5\text{ \AA}$ | $\qty{1}{\angstrom} = \qty{0.1}{\nano\meter} = \qty{1e-10}{\meter}$. |
| **Time / Latency** | $\mathsf{T}$ | `\qty{50}{\milli\second}`, `\qty{2.5}{\second}`, `\qty{15}{\nano\second}` | $50\text{ ms}$, $2.5\text{ s}$, $15\text{ ns}$ | Use non-breaking spaces for time duration. |
| **Velocity / Speed** | $\mathsf{L}\mathsf{T}^{-1}$ | `\qty{3.5}{\kilo\meter\per\second}`, `\qty{12.4}{\meter\per\second}` | $3.5\text{ km s}^{-1}$, $12.4\text{ m s}^{-1}$ | Phase/group velocity in wave propagation. |
| **Acceleration** | $\mathsf{L}\mathsf{T}^{-2}$ | `\qty{9.81}{\meter\per\second\squared}` | $9.81\text{ m s}^{-2}$ | Gravitational acceleration at Earth's surface. |
| **Gravitational Acceleration (Gal)** | $\mathsf{L}\mathsf{T}^{-2}$ | `\qty{980.665}{\gal}`, `\qty{15}{\milli\gal}`, `\qty{20}{\micro\gal}` | $980.665\text{ Gal}$, $15\text{ mGal}$, $20\text{ }\mu\text{Gal}$ | $\qty{1}{\gal} = \qty{1e-2}{\meter\per\second\squared}$; standard in gravimetry. |
| **Gravity Gradient (Eötvös)** | $\mathsf{T}^{-2}$ | `\qty{10}{\eotvos}` | $10\text{ E}$ | $\qty{1}{\eotvos} = \qty{1e-9}{\per\second\squared}$; gravity gradiometry. |
| **Mass & Density** | $\mathsf{M}$, $\mathsf{M}\mathsf{L}^{-3}$ | `\qty{70}{\kilo\gram}`, `\qty{2670}{\kilo\gram\per\meter\cubed}` | $70\text{ kg}$, $2670\text{ kg m}^{-3}$ | Crustal rock density $\rho_0 \approx \qty{2.67}{\gram\per\centi\meter\cubed}$. |
| **Linear Momentum** | $\mathsf{M}\mathsf{L}\mathsf{T}^{-1}$ | `\qty{45}{\kilo\gram\meter\per\second}` or `\qty{45}{\newton\second}` | $45\text{ kg m s}^{-1}$, $45\text{ N s}$ | Impulse-momentum equivalence. |
| **Angular Momentum / Action** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-1}$ | `\qty{1.054e-34}{\joule\second}` | $1.054 \times 10^{-34}\text{ J s}$ | Reduced Planck constant $\hbar = h / (2\pi)$. |
| **Force / Weight** | $\mathsf{M}\mathsf{L}\mathsf{T}^{-2}$ | `\qty{150}{\kilo\newton}`, `\qty{12.5}{\newton}`, `\qty{2.4}{\mega\newton}` | $150\text{ kN}$, $12.5\text{ N}$, $2.4\text{ MN}$ | $\unit{\newton} = \unit{\kilo\gram\meter\per\second\squared}$. |
| **Torque / Moment of Force** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-2}$ | `\qty{250}{\newton\meter}` | $250\text{ N m}$ | Distinct from energy ($\unit{\joule}$); torque is a pseudo-vector. |
| **Energy / Work / Heat** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-2}$ | `\qty{1.2e6}{\joule}`, `\qty{450}{\kilo\joule}`, `\qty{3.5}{\electronvolt}` | $1.2 \times 10^6\text{ J}$, $450\text{ kJ}$, $3.5\text{ eV}$ | $\qty{1}{\electronvolt} \approx \qty{1.60218e-19}{\joule}$. |
| **Power / Radiant Flux** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-3}$ | `\qty{1.5}{\mega\watt}`, `\qty{250}{\kilo\watt}`, `\qty{50}{\milli\watt}` | $1.5\text{ MW}$, $250\text{ kW}$, $50\text{ mW}$ | $\unit{\watt} = \unit{\joule\per\second}$. |

#### 2. Continuum Mechanics, Rheology, Geophysics & Seismology
| Physical Quantity | Dimensional Formula | Canonical `siunitx` Macro | Rendered Output | Contextual Notes & Equivalences |
| :--- | :--- | :--- | :--- | :--- |
| **Stress & Pressure** | $\mathsf{M}\mathsf{L}^{-1}\mathsf{T}^{-2}$ | `\qty{50}{\mega\pascal}`, `\qty{30}{\giga\pascal}`, `\qty{1.013}{\bar}` | $50\text{ MPa}$, $30\text{ GPa}$, $1.013\text{ bar}$ | $\unit{\pascal} = \unit{\newton\per\meter\squared}$; $\qty{1}{\bar} = \qty{1e5}{\pascal}$. |
| **Elastic Moduli ($E, K, \mu$)** | $\mathsf{M}\mathsf{L}^{-1}\mathsf{T}^{-2}$ | `\qty{70}{\giga\pascal}`, `\qty{32}{\giga\pascal}` | $70\text{ GPa}$, $32\text{ GPa}$ | Young's modulus, Bulk modulus, Shear modulus. |
| **Strain / Microstrain** | Dimensionless | `\qty{150}{\micro\strain}` or `\qty{150e-6}{\meter\per\meter}` | $150\text{ }\mu\varepsilon$, $150 \times 10^{-6}\text{ m m}^{-1}$ | $\qty{1}{\micro\strain} = 10^{-6}$; tectonic deformation strain. |
| **Strain Rate** | $\mathsf{T}^{-1}$ | `\qty{1e-14}{\per\second}`, `\qty{25}{\nano\strain\per\year}` | $10^{-14}\text{ s}^{-1}$, $25\text{ n}\varepsilon\text{ a}^{-1}$ | Tectonic plate boundary deformation rate. |
| **Seismic Moment ($M_0$)** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-2}$ | `\qty{3.5e18}{\newton\meter}` | $3.5 \times 10^{18}\text{ N m}$ | $M_{\mathrm{w}} = \frac{2}{3}\log_{10}(M_0 / [\unit{\newton\meter}]) - 6.07$. |
| **Stress Drop ($\Delta \sigma$)** | $\mathsf{M}\mathsf{L}^{-1}\mathsf{T}^{-2}$ | `\qty{3.5}{\mega\pascal}`, `\qty{35}{\bar}` | $3.5\text{ MPa}$, $35\text{ bar}$ | Earthquake rupture dynamic/static stress drop. |
| **Dynamic Viscosity ($\eta, \mu$)** | $\mathsf{M}\mathsf{L}^{-1}\mathsf{T}^{-1}$ | `\qty{1.5}{\pascal\second}`, `\qty{1.0}{\milli\pascal\second}` | $1.5\text{ Pa s}$, $1.0\text{ mPa s}$ | $\qty{1}{\centi\poise} = \qty{1}{\milli\pascal\second}$ (water at $\qty{20}{\degreeCelsius}$). |
| **Kinematic Viscosity ($\nu$)** | $\mathsf{L}^2\mathsf{T}^{-1}$ | `\qty{1.2e-6}{\meter\squared\per\second}` | $1.2 \times 10^{-6}\text{ m}^2\text{ s}^{-1}$ | $\nu = \eta / \rho$; $\qty{1}{\stokes} = \qty{1e-4}{\meter\squared\per\second}$. |
| **Hydraulic Conductivity ($K$)** | $\mathsf{L}\mathsf{T}^{-1}$ | `\qty{1.5e-5}{\meter\per\second}` | $1.5 \times 10^{-5}\text{ m s}^{-1}$ | Darcy seepage velocity in porous media. |
| **Permeability ($k$)** | $\mathsf{L}^2$ | `\qty{1e-13}{\meter\squared}`, `\qty{100}{\milli\darcy}` | $10^{-13}\text{ m}^2$, $100\text{ mD}$ | $\qty{1}{\darcy} \approx \qty{0.986923e-12}{\meter\squared}$. |

#### 3. Thermodynamics, Heat Transfer & Statistical Mechanics
| Physical Quantity | Dimensional Formula | Canonical `siunitx` Macro | Rendered Output | Contextual Notes & Equivalences |
| :--- | :--- | :--- | :--- | :--- |
| **Thermodynamic Temperature** | $\mathsf{\Theta}$ | `\qty{293.15}{\kelvin}`, `\qty{4.2}{\kelvin}` | $293.15\text{ K}$, $4.2\text{ K}$ | Invariant base SI unit; never write $^\circ\mathrm{K}$. |
| **Celsius Temperature** | $\mathsf{\Theta}$ | `\qty{21.5}{\degreeCelsius}`, `\qty{-10.2}{\degreeCelsius}` | $21.5\text{ }^\circ\text{C}$, $-10.2\text{ }^\circ\text{C}$ | $T / [\unit{\kelvin}] = \theta / [\unit{\degreeCelsius}] + 273.15$. |
| **Specific Heat Capacity ($c$)** | $\mathsf{L}^2\mathsf{T}^{-2}\mathsf{\Theta}^{-1}$ | `\qty{4184}{\joule\per\kilo\gram\per\kelvin}` | $4184\text{ J kg}^{-1}\text{ K}^{-1}$ | Specific isobaric heat capacity $c_p$. |
| **Molar Heat Capacity ($C_{\mathrm{m}}$)**| $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-2}\mathsf{\Theta}^{-1}\mathsf{N}^{-1}$ | `\qty{25.1}{\joule\per\mole\per\kelvin}` | $25.1\text{ J mol}^{-1}\text{ K}^{-1}$ | Dulong-Petit high-temperature limit $3R$. |
| **Thermal Conductivity ($k$)** | $\mathsf{M}\mathsf{L}\mathsf{T}^{-3}\mathsf{\Theta}^{-1}$ | `\qty{2.5}{\watt\per\meter\per\kelvin}` | $2.5\text{ W m}^{-1}\text{ K}^{-1}$ | Fourier thermal conduction equation $\mathbf{q} = -k \nabla T$. |
| **Heat Flux Density ($q$)** | $\mathsf{M}\mathsf{T}^{-3}$ | `\qty{65}{\milli\watt\per\meter\squared}` | $65\text{ mW m}^{-2}$ | Terrestrial surface heat flow. |
| **Entropy / Boltzmann Constant**| $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-2}\mathsf{\Theta}^{-1}$ | `\qty{1.380649e-23}{\joule\per\kelvin}` | $1.380649 \times 10^{-23}\text{ J K}^{-1}$ | Fundamental Boltzmann constant $k_{\mathrm{B}}$. |
| **Molar Gas Constant ($R$)** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-2}\mathsf{\Theta}^{-1}\mathsf{N}^{-1}$ | `\qty{8.314462618}{\joule\per\mole\per\kelvin}` | $8.314462618\text{ J mol}^{-1}\text{ K}^{-1}$| $R = N_{\mathrm{A}} k_{\mathrm{B}}$. |
| **Latent Heat / Specific Enthalpy** | $\mathsf{L}^2\mathsf{T}^{-2}$ | `\qty{334}{\kilo\joule\per\kilo\gram}` | $334\text{ kJ kg}^{-1}$ | Enthalpy of fusion for water ice. |

#### 4. Electromagnetism, Optics & Electrodynamics
| Physical Quantity | Dimensional Formula | Canonical `siunitx` Macro | Rendered Output | Contextual Notes & Equivalences |
| :--- | :--- | :--- | :--- | :--- |
| **Electric Charge ($Q$)** | $\mathsf{I}\mathsf{T}$ | `\qty{1.602e-19}{\coulomb}`, `\qty{4.5}{\micro\coulomb}` | $1.602 \times 10^{-19}\text{ C}$, $4.5\text{ }\mu\text{C}$ | $\unit{\coulomb} = \unit{\ampere\second}$. |
| **Electric Potential ($V, \phi$)** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-3}\mathsf{I}^{-1}$ | `\qty{230}{\volt}`, `\qty{15}{\milli\volt}`, `\qty{50}{\micro\volt}` | $230\text{ V}$, $15\text{ mV}$, $50\text{ }\mu\text{V}$ | $\unit{\volt} = \unit{\joule\per\coulomb} = \unit{\watt\per\ampere}$. |
| **Electric Field Strength ($\mathbf{E}$)** | $\mathsf{M}\mathsf{L}\mathsf{T}^{-3}\mathsf{I}^{-1}$ | `\qty{3.0e6}{\volt\per\meter}`, `\qty{15}{\newton\per\coulomb}`| $3.0 \times 10^6\text{ V m}^{-1}$, $15\text{ N C}^{-1}$| Breakdown field strength in air. |
| **Displacement Field ($\mathbf{D}$)** | $\mathsf{L}^{-2}\mathsf{I}\mathsf{T}$ | `\qty{2.5}{\micro\coulomb\per\meter\squared}` | $2.5\text{ }\mu\text{C m}^{-2}$ | Gauss's law $\nabla \cdot \mathbf{D} = \rho_{\mathrm{free}}$. |
| **Capacitance ($C$)** | $\mathsf{M}^{-1}\mathsf{L}^{-2}\mathsf{T}^4\mathsf{I}^2$ | `\qty{100}{\micro\farad}`, `\qty{22}{\pico\farad}` | $100\text{ }\mu\text{F}$, $22\text{ pF}$ | $\unit{\farad} = \unit{\coulomb\per\volt}$. |
| **Permittivity ($\varepsilon$)** | $\mathsf{M}^{-1}\mathsf{L}^{-3}\mathsf{T}^4\mathsf{I}^2$ | `\qty{8.854e-12}{\farad\per\meter}` | $8.854 \times 10^{-12}\text{ F m}^{-1}$ | Vacuum permittivity $\varepsilon_0 = 1 / (\mu_0 c^2)$. |
| **Resistance & Impedance ($R, Z$)** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-3}\mathsf{I}^{-2}$ | `\qty{50}{\ohm}`, `\qty{10}{\kilo\ohm}`, `\qty{2.2}{\mega\ohm}` | $50\text{ }\Omega$, $10\text{ k}\Omega$, $2.2\text{ M}\Omega$ | $\unit{\ohm} = \unit{\volt\per\ampere}$. |
| **Electrical Resistivity ($\rho_{\mathrm{e}}$)** | $\mathsf{M}\mathsf{L}^3\mathsf{T}^{-3}\mathsf{I}^{-2}$ | `\qty{150}{\ohm\meter}` | $150\text{ }\Omega\text{ m}$ | Apparent resistivity in geoelectrical sounding. |
| **Conductance ($G$)** | $\mathsf{M}^{-1}\mathsf{L}^{-2}\mathsf{T}^3\mathsf{I}^2$ | `\qty{20}{\milli\siemens}` | $20\text{ mS}$ | $\unit{\siemens} = \unit{\ohm}^{-1}$. |
| **Electrical Conductivity ($\sigma_{\mathrm{e}}$)** | $\mathsf{M}^{-1}\mathsf{L}^{-3}\mathsf{T}^3\mathsf{I}^2$ | `\qty{5.8e7}{\siemens\per\meter}` | $5.8 \times 10^7\text{ S m}^{-1}$ | Copper conductivity $\sigma_{\mathrm{Cu}}$. |
| **Magnetic Flux Density ($\mathbf{B}$)** | $\mathsf{M}\mathsf{T}^{-2}\mathsf{I}^{-1}$ | `\qty{1.5}{\tesla}`, `\qty{45}{\micro\tesla}`, `\qty{45000}{\nano\tesla}` | $1.5\text{ T}$, $45\text{ }\mu\text{T}$, $45\,000\text{ nT}$ | Earth's field $\approx \qtyrange{30}{60}{\micro\tesla}$; $\qty{1}{\gauss} = \qty{1e-4}{\tesla}$. |
| **Magnetic Field Strength ($\mathbf{H}$)** | $\mathsf{L}^{-1}\mathsf{I}$ | `\qty{40}{\ampere\per\meter}` | $40\text{ A m}^{-1}$ | $\qty{1}{\oersted} = \frac{1000}{4\pi}\unit{\ampere\per\meter} \approx \qty{79.577}{\ampere\per\meter}$. |
| **Magnetic Flux ($\Phi$)** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-2}\mathsf{I}^{-1}$ | `\qty{2.07e-15}{\weber}` | $2.07 \times 10^{-15}\text{ Wb}$ | Flux quantum $\Phi_0 = h / (2e)$; $\unit{\weber} = \unit{\volt\second}$. |
| **Inductance ($L, M$)** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-2}\mathsf{I}^{-2}$ | `\qty{10}{\milli\henry}`, `\qty{2.5}{\micro\henry}` | $10\text{ mH}$, $2.5\text{ }\mu\text{H}$ | $\unit{\henry} = \unit{\weber\per\ampere}$. |
| **Permeability ($\mu$)** | $\mathsf{M}\mathsf{L}\mathsf{T}^{-2}\mathsf{I}^{-2}$ | `\qty{1.257e-6}{\henry\per\meter}` | $1.257 \times 10^{-6}\text{ H m}^{-1}$ | Vacuum permeability $\mu_0 = 4\pi \times 10^{-7}\unit{\newton\per\ampere\squared}$. |
| **Optical Wavelength ($\lambda$)** | $\mathsf{L}$ | `\qty{632.8}{\nano\meter}`, `\qty{1550}{\nano\meter}` | $632.8\text{ nm}$, $1550\text{ nm}$ | Helium-neon laser / telecom IR wavelength. |
| **Spectroscopic Wavenumber ($\tilde{\nu}$)** | $\mathsf{L}^{-1}$ | `\qty{1600}{\per\centi\meter}` | $1600\text{ cm}^{-1}$ | $\tilde{\nu} = 1 / \lambda$; FTIR infrared spectroscopy. |
| **Illuminance / Luminous Flux** | $\mathsf{J}\mathsf{L}^{-2}$, $\mathsf{J}$ | `\qty{500}{\lux}`, `\qty{1200}{\lumen}` | $500\text{ lx}$, $1200\text{ lm}$ | $\unit{\lux} = \unit{\lumen\per\meter\squared}$. |

#### 5. Nuclear, Particle & Atomic Physics
| Physical Quantity | Dimensional Formula | Canonical `siunitx` Macro | Rendered Output | Contextual Notes & Equivalences |
| :--- | :--- | :--- | :--- | :--- |
| **Radioactive Activity ($A$)** | $\mathsf{T}^{-1}$ | `\qty{37}{\kilo\becquerel}`, `\qty{1.5}{\mega\becquerel}` | $37\text{ kBq}$, $1.5\text{ MBq}$ | $\unit{\becquerel} = \unit{\per\second}$; $\qty{1}{\curie} = \qty{3.7e10}{\becquerel}$. |
| **Absorbed Dose & Kerma ($D$)** | $\mathsf{L}^2\mathsf{T}^{-2}$ | `\qty{2.0}{\gray}`, `\qty{50}{\milli\gray}` | $2.0\text{ Gy}$, $50\text{ mGy}$ | $\unit{\gray} = \unit{\joule\per\kilo\gram}$; $\qty{100}{\rad} = \qty{1}{\gray}$. |
| **Equivalent / Effective Dose ($H$)** | $\mathsf{L}^2\mathsf{T}^{-2}$ | `\qty{2.4}{\milli\sievert\per\year}`, `\qty{10}{\micro\sievert}`| $2.4\text{ mSv a}^{-1}$, $10\text{ }\mu\text{Sv}$ | Biological dose equivalent; $\qty{100}{\rem} = \qty{1}{\sievert}$. |
| **Nuclear Cross-Section ($\sigma_{\mathrm{n}}$)**| $\mathsf{L}^2$ | `\qty{2.5}{\barn}`, `\qty{150}{\milli\barn}` | $2.5\text{ b}$, $150\text{ mb}$ | $\qty{1}{\barn} \equiv \qty{1e-28}{\meter\squared} = \qty{100}{\femto\meter\squared}$. |
| **Particle Rest Mass / Energy** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-2}$ | `\qty{125.25}{\giga\electronvolt}`, `\qty{938.27}{\mega\electronvolt}` | $125.25\text{ GeV}$, $938.27\text{ MeV}$ | Higgs boson and proton rest energies ($E_0 = m c^2$). |

#### 6. Astrophysics, Astronomy & Cosmology
| Physical Quantity | Dimensional Formula | Canonical `siunitx` Macro | Rendered Output | Contextual Notes & Equivalences |
| :--- | :--- | :--- | :--- | :--- |
| **Astronomical Unit ($\mathrm{au}$)** | $\mathsf{L}$ | `\qty{1.0}{\astronomicalunit}` | $1.0\text{ au}$ | Mean Earth-Sun distance $\equiv \qty{149597870700}{\meter}$. |
| **Light-Year ($\mathrm{ly}$)** | $\mathsf{L}$ | `\qty{4.24}{\lightyear}` | $4.24\text{ ly}$ | Distance to Proxima Centauri $\approx \qty{9.461e15}{\meter}$. |
| **Parsec ($\mathrm{pc}$)** | $\mathsf{L}$ | `\qty{3.26}{\parsec}`, `\qty{8.5}{\kilo\parsec}`, `\qty{2.5}{\mega\parsec}` | $3.26\text{ pc}$, $8.5\text{ kpc}$, $2.5\text{ Mpc}$ | $\qty{1}{\parsec} \equiv \frac{\qty{1}{\astronomicalunit}}{\tan(\ang{;;1})} \approx \qty{3.0857e16}{\meter}$. |
| **Angular Diameter / Parallax** | Dimensionless | `\qty{1.5}{\arcsecond}`, `\qty{50}{\milli\arcsecond}` | $1.5\text{ arcsec}$, $50\text{ mas}$ | Stellar astrometry resolution ($1'' = \frac{1}{3600}^\circ$). |
| **Spectral Flux Density ($S_\nu$)** | $\mathsf{M}\mathsf{T}^{-2}$ | `\qty{1.5}{\jansky}`, `\qty{250}{\micro\jansky}` | $1.5\text{ Jy}$, $250\text{ }\mu\text{Jy}$ | Radio astronomy; $\qty{1}{\jansky} \equiv \qty{1e-26}{\watt\per\meter\squared\per\hertz}$. |
| **Solar Mass ($M_\odot$)** | $\mathsf{M}$ | `\qty{1.44}{M_\odot}` | $1.44\text{ }M_\odot$ | Chandrasekhar mass limit; $M_\odot \approx \qty{1.9884e30}{\kilo\gram}$. |
| **Solar Luminosity ($L_\odot$)** | $\mathsf{M}\mathsf{L}^2\mathsf{T}^{-3}$ | `\qty{1.0}{L_\odot}` | $1.0\text{ }L_\odot$ | $L_\odot \approx \qty{3.828e26}{\watt}$. |
| **Hubble Parameter ($H_0$)** | $\mathsf{T}^{-1}$ | `\qty{70}{\kilo\meter\per\second\per\mega\parsec}` | $70\text{ km s}^{-1}\text{ Mpc}^{-1}$ | Cosmic expansion rate. |

#### 7. Dimensionless Ratios, Transport Numbers & Characteristic Groups
Typeset characteristic dimensionless numbers in fluid dynamics and transport phenomena in upright Roman two-letter symbols:

| Dimensionless Group | Standard Symbol | Mathematical Definition | Physical Interpretation |
| :--- | :---: | :--- | :--- |
| **Reynolds Number** | $\mathrm{Re}$ | $\mathrm{Re} = \frac{\rho v L}{\mu} = \frac{v L}{\nu}$ | Ratio of inertial forces to viscous forces in fluid flow. |
| **Mach Number** | $\mathrm{Ma}$ | $\mathrm{Ma} = \frac{v}{c_{\mathrm{s}}}$ | Ratio of flow velocity to local acoustic phase speed. |
| **Rayleigh Number** | $\mathrm{Ra}$ | $\mathrm{Ra} = \frac{g \beta \Delta T L^3}{\nu \alpha}$ | Ratio of buoyancy-driven thermal convection to thermal diffusion. |
| **Prandtl Number** | $\mathrm{Pr}$ | $\mathrm{Pr} = \frac{\nu}{\alpha} = \frac{\mu c_p}{k}$ | Ratio of kinematic momentum diffusivity to thermal diffusivity. |
| **Knudsen Number** | $\mathrm{Kn}$ | $\mathrm{Kn} = \frac{\lambda_{\mathrm{mfp}}}{L}$ | Ratio of molecular mean free path to characteristic physical length scale. |
| **Froude Number** | $\mathrm{Fr}$ | $\mathrm{Fr} = \frac{v}{\sqrt{g L}}$ | Ratio of inertial forces to gravitational body forces. |
| **Péclet Number** | $\mathrm{Pe}$ | $\mathrm{Pe} = \frac{v L}{D} = \mathrm{Re} \cdot \mathrm{Sc}$ | Ratio of advective transport rate to diffusive transport rate. |
| **Nusselt Number** | $\mathrm{Nu}$ | $\mathrm{Nu} = \frac{h L}{k}$ | Ratio of convective heat transfer to conductive heat transfer. |
| **Decibels (Power / Voltage SNR)**| $\unit{\decibel}$ | `\qty{20}{\decibel}` | Logarithmic power ratio: $10 \log_{10}(P_1 / P_0)$ or $20 \log_{10}(V_1 / V_0)$. |
| **Parts Per Notation** | $\unit{\ppm}$, $\unit{\ppb}$ | `\qty{25}{\ppm}`, `\qty{10}{\ppb}` | Trace chemical concentrations ($\qty{1}{\ppm} = 10^{-6}, \qty{1}{\ppb} = 10^{-9}$). |

---

### D. Advanced `siunitx` v3 Formatting Configurations & Syntax Safeguards

#### 1. Unit Product and Quotient Formatting
Enforce explicit compound unit representations in the LaTeX preamble or localized scope:
```latex
\sisetup{
  inter-unit-product = \cdot , % Uses centered dot (N\cdot m) instead of thin space
  per-mode           = power , % Formats as m s^{-1}; use 'symbol' for m/s or 'fraction' for \frac{m}{s}
  uncertainty-mode   = separate % Formats \qty{1.23 \pm 0.04}{\meter} as (1.23 \pm 0.04) m
}
```

#### 2. Table Column Decimal Alignment (`S` Descriptors)
In publication tables, align floating-point columns with `S` descriptors. Always enclose text headers and non-numeric labels in curly braces `{...}`:
```latex
\begin{tabular}{l S[table-format=3.2] S[table-format=+1.3e-2]}
  \toprule
  {Geological Layer} & {P-Wave Velocity [\unit{\kilo\meter\per\second}]} & {Deviatoric Stress [\unit{\mega\pascal}]} \\
  \midrule
  Upper Sediments    & 2.45                                              & 1.20e-1                                  \\
  Crystalline Crust  & 6.10                                              & 1.45e1                                   \\
  Upper Mantle (Pn)  & 8.15                                              & 3.20e2                                   \\
  \bottomrule
\end{tabular}
```

#### 3. Pure Number Scientific Formatting (`\num`)
Use `\num` for formatted numbers with automatic thousands grouping and standard exponent representation:
- `\num{10000}` renders as $10\,000$ (with thin grouping space).
- `\num{6.02214076e23}` renders as $6.02214076 \times 10^{23}$.
- `\num{-9.81}` renders with an authentic mathematical minus sign ($-9.81$), avoiding ASCII hyphen hyphens.

---

### E. Package Compatibility Safeguard (`\usepackage{physics}`)
> [!IMPORTANT]
> When compiling in legacy repositories or third-party templates where `\usepackage{physics}` is loaded, the macro `\qty(...)` is overloaded by the `physics` package as a flexible delimiter resizer (`\qty(x) \equiv \left( x \right)`).
> In manuscripts loading `physics.sty`:
> - Use the legacy `siunitx` v2 syntax: `\SI{value}{unit}` for quantities (e.g., `\SI{15}{\kilo\meter}`).
> - Use `\si{unit}` for standalone units (e.g., `\si{\kilo\meter\per\second}`).
> - Alternatively, configure `siunitx` with `\usepackage[load=named]{siunitx}` or load `siunitx` before `physics` and undefine `\qty`:
>   ```latex
>   \usepackage{siunitx}
>   \let\qty\SI % Alias \qty back to \SI if physics is subsequently loaded
>   ```

