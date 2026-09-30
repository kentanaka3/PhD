# Flowchart, Pseudocode & Mermaid Reference

> Comprehensive technical reference for embedding in agent specification documents.
> Sources: ANSI X3.5, ISO 5807, IEEE, Mermaid.js v11.3+ documentation.

---

## 1. Flowchart Diagram Types

### 1.1 Core Flowchart Types (ANSI/ISO Classification)

| Type | Definition | Typical Use Case | Distinguishing Feature |
|:-----|:-----------|:-----------------|:-----------------------|
| **Document Flowchart** | Illustrates the flow of physical or electronic documents through an organization—tracking how forms, reports, and records move between departments. | Auditing, accounting, internal controls, compliance review. | Focus is on **document routing and custody**, not data transformation or program logic. |
| **Data Flowchart** (DFD) | Maps how data moves through a system, showinginputs, outputs, data stores, and transformations between processes and external entities. | Requirements analysis, system design, information architecture. | Focus is on **data transformation and storage**; no control-flow sequencing—processes may execute in any order. |
| **System Flowchart** | Provides a high-level view of an entire system atthe physical or resource level, showing relationships among hardware, software, data files, and human actors. | System architecture design, infrastructure planning, integration analysis. | Focus is on **physical components and resource allocation**, not individual program logic or document custody.|
| **Program Flowchart** | Details the step-by-step logic and control flow within a single program or algorithm, including decisions, loops, and subroutine calls. | Algorithm design, coding, debugging, code review. | Focus ison **internal control flow** (sequence, selection, iteration) within one executable unit. |
| **Reversible Flowchart** | A formal computational model where every stepis locally invertible, ensuring the input can be perfectly reconstructed from the output without information loss. | Reversible computing, quantum computing, adiabatic circuit design; languages like Janus, R-CORE, R-WHILE. |Every operation is **bijective** (one-to-one); the flowchart can be executed forwards and backwards. Governed by the Structured Reversible Program Theorem. |

### 1.2 Additional Well-Established Flowchart Types

| Type | Definition | Typical Use Case | Distinguishing Feature |
|:-----|:-----------|:-----------------|:-----------------------|
| **Swim Lane / Cross-Functional Flowchart** | A flowchart divided into parallel horizontal or vertical lanes, each representing a department, role, or system. Process steps are placed in the lane of the responsible party. |Cross-departmental process mapping, accountability analysis, handoff identification. | Adds a **responsibility dimension**; visually answers "who does what." |
| **Workflow Flowchart** | A general-purpose diagram mapping the sequence of tasks required to complete a business process from start to finish. | SOPs, employee training, operational documentation. | Emphasizes **task sequencing and completion criteria** rather than data or control logic. |
| **Process Flowchart** | A step-by-step map of all steps and decisions ina process, often used in quality management (Six Sigma, Lean). | Process improvement, bottleneck identification, quality control. | Typically includes **measurement points, decision gates, and rework loops**. |
| **Event-Driven Process Chain (EPC)** | A modeling language showing the logical and chronological relationship between events (states) and functions(activities), connected by AND/OR/XOR operators. | ERP implementations (SAP), enterprise process analysis, business process reengineering. | Uses **explicit logical connectors** (∧, ∨, ⊕) between alternating events and functions. |
| **SDL Diagram** (Specification and Description Language) | A formal, standardized graphical language (ITU-T Z.100) for specifying the behavior of reactive, real-time, and distributed systems as communicating state machines. | Telecommunications protocols, automotive systems, aviation, medical devices. | **Formally executable**; describes systems as state machines exchanging discrete signals. |
| **Signal / Event Flow Diagram** | Shows the flow of signals or events through a system, focusing on triggers, handlers, and event propagation. | Real-time systems, interrupt-driven architectures, UI event handling. | Focusis on **asynchronous event/signal propagation** rather than sequential control flow. |

---

## 2. ANSI/ISO Standard Flowchart Symbols

> Based on **ANSI X3.5** (predecessor) and **ISO 5807:1985** — *Information processing — Documentation symbols and conventions for data, program and system flowcharts, program network charts and system resources charts.*

### 2.1 Terminal & Flow Symbols

| Symbol Name | Shape | Represents | When to Use |
|:------------|:------|:-----------|:------------|
| **Terminal / Terminator** | Oval (rounded rectangle, "stadium") | The start or end point of a process or program. | Mark the single entry and exit points of any flowchart. |
| **Flow Line** | Arrow (solid line with arrowhead) | Direction and sequence of process flow. | Connect every symbol; standard direction is top→bottom or left→right. Arrowheads are mandatory when flow reverses direction. |
| **Connector** (On-page) | Small circle | Junction point linking separateparts of a flowchart on the same page. Uses a matching label (letter/number). | Reduce crossing flow lines; connect distant parts of a large diagram on one page. |
| **Off-page Connector** | Pentagon (home-plate shape, rectangle with pointed bottom) | Indicates the flow continues on a different page or sheet. Contains a cross-reference label. | Link multi-page flowcharts. |

### 2.2 Process & Operation Symbols

| Symbol Name | Shape | Represents | When to Use |
|:------------|:------|:-----------|:------------|
| **Process** | Rectangle | A single action, operation, or computational step (e.g., `x ← x + 1`). | Any defined operation that transforms data or changes system state. |
| **Predefined Process / Subroutine** | Rectangle with double vertical bars on left and right sides | A named subprocess, function, or module definedand documented elsewhere. | Invoking a reusable routine; the detail is in a separate flowchart. |
| **Preparation** | Hexagon | An initialization, setup, or loop-control step (e.g., `Set i = 0` or `Initialize counter`). | Loop variable initialization, clearing buffers, setting flags before a loop or process block. |
| **Manual Operation** | Trapezoid (wider at top, narrower at bottom) | A process step performed manually by a human, not by a machine. | Data entry by hand, physical inspection, manual approval steps. |
| **Parallel Mode** | Two horizontal bars (synchronization bars) | Beginning or end of two or more simultaneous (parallel) operations. | Fork/join ofconcurrent processes; always used in matched pairs. |

### 2.3 Decision & Branching Symbols

| Symbol Name | Shape | Represents | When to Use |
|:------------|:------|:-----------|:------------|
| **Decision** | Diamond (rhombus) | A conditional branch point: Yes/No, True/False, or multi-way selection. Outgoing flows are labeled with conditions. | Any `if`, `switch`, or conditional test. |
| **Merge** | Inverted triangle (triangle pointing downward) | Convergencepoint where multiple flow paths combine into a single path (no decision logic). | Rejoining branches after a decision or parallel split. |
| **Extract** | Upward-pointing triangle | Splitting a single flow into multiple paths, or selecting/filtering a subset from a data set. | Data filtering, subset extraction, one-to-many routing. |

### 2.4 Input/Output & Data Symbols

| Symbol Name | Shape | Represents | When to Use |
|:------------|:------|:-----------|:------------|
| **Input/Output (Data)** | Parallelogram | Generic data entering or leaving the system (read, write, display, print). | Any I/O operation not covered by a more specific symbol. |
| **Document** | Rectangle with a wavy bottom edge | A single document, report, or printed output. | Output to paper, PDF generation, form submission. |
| **Multi-Document** | Stacked documents (overlapping wavy-bottom rectangles) | A set or batch of documents. | Batch reports, multi-page output, document packets. |
| **Manual Input** | Parallelogram/rectangle with sloped top edge (tilted top-right) | Data entered manually by a human at the time of processing (e.g., keyboard entry). | Prompts for user input, form filling, command-line entry. |
| **Display** | Curved trapezoid (rectangle with one curved side, resembling a CRT screen) | Information displayed to a user on a screen or monitor. | Screen output, dashboard display, console messages. |

### 2.5 Storage Symbols

| Symbol Name | Shape | Represents | When to Use |
|:------------|:------|:-----------|:------------|
| **Stored Data** (generic) | Horizontal cylinder or bow-tie rectangle (rectangle with one curved side) | Data stored in any medium (generic, non-specific storage). | General-purpose data persistence when the medium is unspecified. |
| **Database** | Vertical cylinder | Data stored in a database management system. | SQL/NoSQL database reads or writes. |
| **Internal Storage** | Rectangle with a small square in the upper-left corner ("window pane") | Data stored in main memory (RAM) during program execution. | Temporary buffers, in-memory caches, working variables. |
| **Delay** | Half-D shape (semicircle / "bullet" shape, flat on left, rounded on right) | A waiting period, time delay, or queue in the process. | Approval waits, processing queues, timed pauses, buffering. |

### 2.6 Data Manipulation Symbols

| Symbol Name | Shape | Represents | When to Use |
|:------------|:------|:-----------|:------------|
| **Sort** | Diamond divided horizontally into upper and lower triangles (hourglass) | Arranging a set of items into a defined sequence. | Sorting operations on datasets. |
| **Collate** | Hourglass (two triangles meeting at a point) | Merging andinterleaving two or more ordered sets into one. | Merge-sort operations, combining sorted lists. |

### 2.7 Annotation & Commentary

| Symbol Name | Shape | Represents | When to Use |
|:------------|:------|:-----------|:------------|
| **Annotation / Comment** | Open bracket (square bracket or curly brace) connected to a symbol by a dashed line | Descriptive comment or explanatorynote that does not affect process logic. | Adding clarifications, assumptions, constraints, or references without altering the flow. |

---

## 3. Canonical Mathematical Pseudocode Symbols & Notation

### 3.1 Assignment, Binding & Definitions

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `←` | Assign value | `x ← x + 1` |
| `≝` | Equal by definition | `SNR ≝ 10 × \log_{10}(P_{\text{signal}} / P_{\text{noise}})` |
| `≡` | Identical / Congruent | `hash(x) ≡ 0 \pmod{m}` |
| `x ↦ f(x)` | Element mapping rule | `t ↦ t − t₀` |
| `λx. e` | Lambda abstraction | `filter(λp. p.weight > 0, picks)` |
| `ι` | Definite description / unique selection | `x^* ← ι x . (P(x) ∧ ∀ y : P(y) ⇒ y = x)` |

> **Convention**: Use `←` to distinguish assignment from equality comparison (`=`).

### 3.2 Comparison, Ordering & Distribution

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `=` | Equal to | `if x = 0 ⇒ …` |
| `≠` | Not equal to | `while key ≠ target ⇒ …` |
| `<` | Strictly less than | `if i < n ⇒ …` |
| `>` | Strictly greater than | `if score > threshold ⇒ …` |
| `≤` | Less than or equal to | `∀ i ∈ [1 .. n] : i ≤ n` |
| `≥` | Greater than or equal to | `while count ≥ 0 ⇒ …` |
| `≈` | Approximately equal to | `if residual ≈ 0.0 ⇒ ⟵ ⊤` |
| `∼` | Distributed as / Similarity | `noise ∼ 𝒩(0, σ²)` or `f(n) ∼ g(n)` |
| `∝` | Proportional to | `P(θ \| D) ∝ P(D \| θ) × P(θ)` |
| `≅` | Isomorphic to | `G₁ ≅ G₂` |
| `⊑`, `⊒` | Information ordering / Subsumption | `s₁ ⊑ s₂` |
| `⊥` | Orthogonal / Independent | `X ⊥ Y` |
| `∥` | Parallel | `v₁ ∥ v₂` |

### 3.3 Proof Theory, Logic & Type Judgments

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `⊢` | Turnstile: Syntactic entailment / Type judgment | `Γ ⊢ e : τ` or `pre ⊢ state_valid` |
| `⊨` | Semantic entailment / Model satisfaction | `ℳ ⊨ φ` |
| `⊬`, `⊭` | Negated syntactic / semantic entailment | `Γ ⊬ contradiction` |
| `∧` (`and`) | Logical conjunction | `valid ∧ ¬expired ⇒ …` |
| `∨` (`or`) | Logical disjunction | `failed ∨ timeout ⇒ …` |
| `¬` (`not`) | Logical negation | `¬exists(path) ⇒ …` |
| `⊕` (`xor`) | Exclusive `or` | `a ⊕ b ≝ (a ∨ b) ∧ ¬(a ∧ b)` |
| `⇒` (`implies`) | Logical implication | `valid ⇒ score > 0` |
| `⇔` (`iff`) | Logical equivalence | `converged ⇔ residual < ε` |
| `⊤` (`true`) | Boolean True / Top | `found ← ⊤` |
| `⊥` (`false`) | Boolean False / Bottom / Error / Null | `if err ≠ ⊥ ⇒ fail ⊥` |

### 3.4 Operational Semantics & Transitions

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `⇓` | Big-step evaluation / Natural semantics | `⟨e, σ⟩ ⇓ ⟨v, σ'⟩` |
| `⇑` | Divergence / Non-termination | `⟨e, σ⟩ ⇑` |
| `⟶` | Small-step reduction / Transition | `⟨e, σ⟩ ⟶ ⟨e', σ'⟩` |
| `↠` | Multi-step reduction (reflexive-transitive) | `e ↠ v` |
| `⟵` | Return value / Result assignment | `⟵ manifest` |
| `⤅` | Yield item (generator) | `⤅ next_sample` |
| `⟦·⟧` | Denotational semantics brackets | `⟦program⟧ : State → State` |
| `⫽` | Fallback / Coalescing | `val ← cached ⫽ compute()` |
| `⑂` | Fork / Spawn asynchronous task | `⑂ worker(task)` |
| `⑃` | Join / Synchronize concurrent tasks | `⑃` |

### 3.5 Calculus, Analysis & Differential Operators

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `∫` (`integral`) | Definite / Indefinite integral | `F ∈ AC([a, b], ℝ) ⟺ (∃ F' m-a.e.) ∧ (F' ∈ L¹([a, b], m)) ∧ (∀x ∈ [a, b], F(x) = F(a) + ∫_a^x F'(t) dt) ; a, b ∈ ℝ, a < b` |
| `∬`, `∭` (`integral`) | Double / Triple surface or volume integral | `∭_{Ω} (∇ · 𝐅) d𝑉 = ∯_{∂Ω} (𝐅 · 𝐧) d𝑆 ; Ω ⊂ ℝ³, 𝐅 ∈ C¹(Ω, ℝ³), 𝐧 : ∂Ω → S² ≔ {v ∈ ℝ³ : ‖v‖ = 1} ∧ 𝐧(x) ⟂ T_x(∂Ω), d𝑉 ∈ ℳ(Ω), d𝑆 ∈ ℳ(∂Ω)` |
| `∮` | Contour / Line integral | `∮_{∂Ω} 𝐅 · 𝐓 \mathrm{d}s = ∬_{Ω} (∇ × 𝐅) · 𝐧 d𝑆 ; Ω ⊂ U ⊆ ℝ³, 𝐅 ∈ C¹(U, ℝ³), 𝐧 : Ω → S² ≝ {v ∈ ℝ³ : ‖v‖ = 1} ∧ 𝐧(x) ⟂ T_x Ω, 𝐓 : ∂Ω → S² ∧ 𝐓(x) ∈ T_x(∂Ω) ∧ ‖𝐓‖ = 1, d𝑆 ∈ ℳ(∂Ω), d𝑆 ∈ ℳ(Ω)` |
| `∯` | Surface integral over closed surface | `∯_{∂Ω} 𝐅 · 𝐧 d𝑆` |
| `∂` (`partial`) | Partial derivative / Manifold boundary | `J_{ij} ← \frac{∂r_i}{∂m_j}` |
| `∇` (`grad`) | Nabla / Gradient operator | `∇ : C^k(Ω, ℝ) → C^{k-1}(Ω, ℝⁿ), f ↦ ∑_{i=1}ⁿ (∂_i f) 𝐞_i ; Ω ⊆ ℝⁿ, n ∈ ℕ_{≥ 1}, k ∈ ℕ_{≥ 1} ∪ {∞}` |
| `∇·` (`div`) | Divergence | `∇· : C^k(Ω, ℝⁿ) → C^{k-1}(Ω, ℝ), 𝐅 ↦ ∑_{i=1}ⁿ ∂_i F^i ; Ω ⊆ ℝⁿ, n ∈ ℕ_{≥ 1}, k ∈ ℕ_{≥ 1} ∪ {∞}` |
| `∇×` (`curl`) | Curl / Rotor | `∇× : C^k(Ω, ℝⁿ) → C^{k-1}(Ω, 𝔰𝔬(n)), 𝐅 ↦ ½ (D𝐅 - (D𝐅)ᵀ) ≅ C^{k-1}(Ω, ℝ^{n(n-1)/2}) ; Ω ⊆ ℝⁿ, n ∈ ℕ_{≥ 1}, k ∈ ℕ_{≥ 1} ∪ {∞}` |
| `Δ` | Laplacian operator (`∇²`) / Difference | `Δf ≝ ∇· ∇f` |
| `lim` | Limit | `\lim_{Δt \to 0} \frac{f(t+Δt) − f(t)}{Δt}` |
| `\mathrm{d}` | Differential | `\mathrm{d}t, \, \mathrm{d}x` |
| `\|x\|` | Absolute value | `delta ← \|x − x₀\|` |
| `‖·‖` | Vector, matrix or operator norm | `‖r‖₂ = √{rᵀ r}` |

### 3.6 Algebraic, Tensor & Signal Operators

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `⋆` | Convolution / Kleene star / Dual | `y ← x ⋆ h` or `Σ^⋆` |
| `⊛` | Circular convolution / Cross-correlation | `C_{xy} ← x ⊛ y` |
| `∘` | Function composition | `(f ∘ g)(x) ≝ f(g(x))` |
| `·` | Scalar dot product | `\mathbf{u} · \mathbf{v} ≝ Σ_i u^i v^i` |
| `×` | Vector cross product / Cartesian product | `\mathbf{u} × \mathbf{v} = sgn(det(g)) √\|det(g)\| g^{mi} ε_{ijk} u^j v^k 𝐞_m; \mathbf{u} = u^j 𝐞_j, \mathbf{v} = v^k 𝐞_k, i,j,k,m ∈ {1,2,3}, det(g) ≠ 0` |
| `⊗` | Tensor / Kronecker product | `(A ⊗ B)_{ik,jl} ≝ a_{i,j} b_{k,l}` |
| `⊕` | Direct sum | `V ⊕ W` |
| `⊙` | Hadamard element-wise product | `A ⊙ B ≝ [a_{ij} b_{ij}]` |
| `ᵀ` | Matrix transpose | `Jᵀ r` |
| `A^†` | Moore-Penrose pseudo-inverse / Adjoint | `m ← (Gᵀ G)^{−1} Gᵀ d` |
| `ℱ{·}` | Fourier transform | `\hat{u}(ω) ← ℱ{u(t)}` |
| `ℋ{·}` | Hilbert transform | `u_H(t) ← ℋ{u(t)}` |

### 3.7 Arithmetic, Floor & Ceiling

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `+` | Addition | `sum ← a + b` |
| `−` | Subtraction | `diff ← a − b` |
| `×` | Multiplication | `area ← length × width` |
| `/` | Real division | `mean ← sum / n` |
| `div` | Integer division (quotient) | `q ← a div b` |
| `mod` | Modulo (remainder) | `r ← a mod b` |
| `^` | Exponentiation (`xⁿ`, `x¹`, `x²`, `x³`, etc.) | `e^{iπ} + 1 = 0` |
| `√` | Square root | `rms ← √(sum_sq / n)` |
| `⌊·⌋` | Floor function | `mid ← ⌊(lo + hi) / 2⌋` |
| `⌈·⌉` | Ceiling function | `pages ← ⌈n / page_size⌉` |

### 3.8 Set Theory, Lattice Theory & Aggregations

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `∈` | Element of (membership) | `if x ∈ S ⇒ …` |
| `∉` | Not element of | `if s ∉ R ⇒ …` |
| `⊂` | Proper subset | `A ⊂ Ω` |
| `⊆` | Subset (inclusive) | `A ⊆ 𝒰` |
| `∪` | Union | `C ← A ∪ B` |
| `∩` | Intersection | `overlap ← A ∩ B` |
| `\` | Set difference / relative complement | `unprocessed ← all \ processed` |
| `Δ` | Symmetric difference | `diff ← set_a Δ set_b` |
| `∅` | Empty set | `if candidates = ∅ ⇒ ⟵ ⊥` |
| `\|S\|` | Set cardinality / collection length | `n ← \|events\|` |
| `𝒫(S)` | Power set | `subsets ← 𝒫(features)` |
| `⊔` | Lattice join / Least upper bound | `lub ← x ⊔ y` |
| `⊓` | Lattice meet / Greatest lower bound | `glb ← x ⊓ y` |
| `Σ` | Summation over bounded index or set | `total ← Σ_{i=1}^{n} a[i]` |
| `Π` | Product over bounded index or set | `prob ← Π_{i=1}^{k} p_i` |
| `min` | Minimum value | `best ← min_{x ∈ S} f(x)` |
| `max` | Maximum value | `peak ← max(a, b)` |
| `argmin` | Argument minimizing the objective | `opt_θ ← argmin_{θ} Loss(θ)` |
| `argmax` | Argument maximizing the objective | `best_c ← argmax_{c} P(c \| x)` |
| `{x ∈ S : P(x)}` | Set comprehension / filtering | `valid ← {x ∈ S : score(x) > θ}` |

### 3.9 Quantifiers

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `∀` | Universal quantifier ("for all" / loop) | `∀ x ∈ S : x > 0` |
| `∃` | Existential quantifier | `if ∃ e ∈ E : e.id = target ⇒ …` |
| `∄` | Negative existential | `if ∄ f ∈ 𝒟 : f.active = ⊤ ⇒ …` |
| `∃!` | Unique existential | `if ∃! master ∈ nodes : master.active = ⊤ ⇒ …` |
| `:` | "such that" / "where" | `∀ s ∈ S : s.active ← ⊤` |

### 3.10 Sequences, Intervals & Special Constants

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| `⟨x₁, x₂, …, xₙ⟩` | Ordered tuple or vector | `p ← ⟨t₀, x₀, y₀, z₀⟩` |
| `A[i]` | Array indexing | `first ← waveforms[1]` |
| `A[i..j]` | Array slice / subsequence | `window ← trace[start..end]` |
| `s₁ ∥ s₂` | String / sequence concatenation | `greeting ← "Hello" ∥ ", " ∥ "World"` |
| `[a, b]` | Closed numerical interval | `freq ∈ [0.5, 20.0]` |
| `(a, b)` | Open numerical interval | `residual ∈ (−1.0, 1.0)` |
| `[a, b)` | Half-open interval | `bin_range ← [t_start, t_end)` |
| `f : X → Y` | Function signature | `locate : Catalog × Model → Hypo` |
| `x ⟶ g ⟶ y` | Pipelined sequential dataflow | `stream ⟶ filter ⟶ picker ⟶ catalog` |
| `∞` | Infinity | `min_cost ← ∞` |
| `NaN` | Not-a-Number (floating-point error) | `if value = NaN ⇒ …` |
| `ε` | Machine epsilon / infinitesimal tolerance | `if \|x_{k+1} − x_k\| > ε ⇒ ⟳` |
| `π`, `e`, `φ`, `i`, `ℏ`, `ℝ`, `ℂ`, `ℕ`, `ℤ` | Mathematical constants | `A ← π × r²` |

### 3.11 Asymptotic Complexity Notation

| Symbol | Meaning | Usage |
|:-------|:--------|:------|
| `O(g(n))` | Upper bound (Big-O) | `Time: O(n log n), Space: O(n)` |
| `Ω(g(n))` | Lower bound (Big-Omega) | `Comparisons: Ω(n log n)` |
| `Θ(g(n))` | Tight asymptotic bound (Big-Theta) | `Lookup: Θ(1) average case` |
| `o(g(n))` | Strict upper bound (Little-o) | `error = o(1) as n → ∞` |

---

## 4. ANSI ↔ Mermaid.js Syntax Mapping

Use **Modern `@{ shape: }` Syntax** (Mermaid v11.3.0+). Use `id@{ shape: <name>, label: "Text" }` for complete ANSI fidelity:

| ANSI/ISO Symbol | Mermaid `shape:` | Semantic Use |
|:----------------|:-----------------|:-------------|
| **Process** | `rect` | Action/Computation/Assignment |
| **Terminal** | `stadium` | Start/End of process |
| **Decision** | `diam` | Conditional branch (`⊤`, `⊥`) |
| **Input/Output** | `lean-r`, `lean-l` | Generic data input/output |
| **Predefined Process** | `fr-rect`, `subroutine` | Named external subprocess |
| **Preparation** | `hex` | Loop initialization / setup |
| **Document** | `doc` | Single document output |
| **Multi-Document** | `docs` | Batch document output |
| **Manual Input** | `sl-rect`, `manual-input` | Human/Console entry |
| **Manual Operation** | `trap-t` | Human-performed task |
| **Display** | `curv-trap` | Screen/console/UI output |
| **Stored Data** | `bow-rect` | Abstract data persistence |
| **Database** | `cyl`, `cylinder` | Relational/Document database |
| **Internal Storage** | `win-pane` | In-memory RAM Buffer/Cache |
| **Direct Access Storage**| `h-cyl` | Disk storage |
| **Delay** | `delay` | Timeout/Sleep/Queue |
| **Connector** | `circle` | On-page junction |
| **Off-page Connector** | `notch-pent` | Cross-page/Cross-module continuation |
| **Merge** | `tri` | Multi-path convergence |
| **Extract** | `flip-tri` | Data/Stream split |
| **Sort / Collate** | `hourglass` | Sort/Merge |
| **Annotation / Comment** | `brace`, `brace-r`, `braces`, `comment` | Explanatory note |
| **Small Circle** | `sm-circ` | Compact junction |
| **Filled Circle** | `f-circ` | Merge/Junction |
| **Double Circle** | `dbl-circ` | Halt/Stop |
| **Framed Circle** | `fr-circ` | Guarded stop |
| **Crossed Circle** | `cross-circ` | Diagnostic summary |
| **Lined Rectangle** | `lin-rect` | Shaded/Composite process |
| **Lined Cylinder** | `lin-cyl` | Structured persistent storage |
| **Lined Document** | `lin-doc` | Structured report |
| **Tagged Document** | `tag-doc` | Versioned document |
| **Tagged Rectangle** | `tag-rect` | Versioned process |
| **Divided Rectangle** | `div-rect` | Partitioned process |
| **Notched Rectangle** | `notch-rect` | Configuration/Raw input |
| **Paper Tape / Flag** | `paper-tape`, `flag` | Telemetry/event marker |
| **Lightning Bolt** | `bolt` | Network/Communication link |
| **Cloud** | `cloud` | Network/Cloud/Cluster |
| **Odd** | `odd` | Irregular/Odd block |
| **Exception / Alert** | `bang` | Exception/Alert |

### 4.1 Quick Decision Guide

```text
Need a shape?
├─ Is it a basic shape (rect, diamond, circle, parallelogram, stadium, hexagon, cylinder)?
│   └─ YES → Use bracket syntax for brevity: id[Label], id{Label}, etc.
├─ Is it a specialized symbol (document, delay, display, hourglass, etc.)?
│   └─ YES → Use @{ shape: <name> } syntax
└─ Not sure?
    └─ Use @{ shape: <name> } — it covers everything and is self-documenting
```

### 4.2 Flow Line Syntax

| Connection Type | Syntax | Example |
|:----------------|:-------|:--------|
| Arrow (solid) | `-->` | `A --> B` |
| Arrow with label | `-->|text|` | `A -->|Yes| B` |
| Thick arrow | `==>` | `A ==> B` |
| Dotted arrow | `-.->` | `A -.-> B` |
| Dotted with label | `-. text .->` | `A -. maybe .-> B` |
| No arrow (link) | `---` | `A --- B` |

---

## References

- **ISO 5807:1985** — Information processing — Documentation symbols and conventions for data, program and system flowcharts, program network charts and system resources charts.
- **ANSI X3.5-1970** — Flowchart Symbols and Their Usage in Information Processing (predecessor to ISO 5807).
- **FIPS PUB 24** (NIST) — Flowchart Symbols and Their Usage in Information Processing.
- **Mermaid.js Documentation** — https://mermaid.js.org/syntax/flowchart.html
- **Yokoyama, Axelsen, Glück (2016)** — "Fundamentals of reversible flowchart languages" — Theoretical Computer Science 611.