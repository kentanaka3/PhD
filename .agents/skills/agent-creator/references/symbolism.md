# Canonical Pseudocode Notation

## 1 Assignment & Definitions

|Symbol|Meaning|Example|
|:---|:---|:---|
|`←`|Assignment|`x ← x+1`|
|`≝`|Definition|`f(x) ≝ x²+2x+1`|
|`=`|Value equality|`count = 0`|
|`≠`|Value inequality|`a ≠ b`|
|`<`, `≤`, `>`, `≥`|Declared ordering|`0 ≤ i < n`|
|`≡`|Identity; criterion locally defined|`σ′.worktree(p) ≡ σ.worktree(p)`|
|`x ↦ f(x)`|Mapping|`t ↦ t−t₀`|
|`λx. e`|Lambda|`filter(λx. x.weight > 0, items)`|
|`ι x ∈ S . P(x)`|Unique selection|`v ← ι x ∈ S . P(x)`|

```text
a,b,m ∈ Integer; m > 0
a ≡ b (mod m) ⇔ ∃ k ∈ Integer : a−b = km
Defined(ι x ∈ S . P(x)) ⇔ ∃! x ∈ S : P(x)
¬Defined(selection) ⇒ Route(Error ⊍ Q)
```

### Local Legend

Entry: minimal legend
extensions: on demand.

```text
τ ≝ task
σ/σ′ ≝ prior/result state
Wτ ≝ targets ∩ permitted paths ∩ authorized paths
Aτ ≝ authorized actions
Vτ ≝ required checks
Q ≝ inquiry
G ≝ guard
A ≝ action
T,E ≝ value/error types
Commands ≝ {if G : A, for x ∈ S : A}
Formulas ≝ properties
Overload ⇒ LocalLegend
Objective ≝ min Characters subject to ExplicitMeaning
Measures ≝ ⟨bytes, characters, model_tokens⟩
```

## 2 Logic & Types

|Symbol|Meaning|Example|
|:---|:---|:---|
|`⊢`|Derivability / typing|`Γ ⊢ φ`; `Γ ⊢ e : T`|
|`⊨`|Entailment / satisfaction|`Γ ⊨ φ`; `ℳ ⊨ φ`|
|`⊬`|Non-derivability|`Γ ⊬ φ`|
|`⊭`|Non-entailment / non-satisfaction|`Γ ⊭ φ`; `ℳ ⊭ φ`|
|`∧` (`and`)|Conjunction|`valid ∧ ¬expired`|
|`∨` (`or`)|Disjunction|`failed ∨ timeout`|
|`¬` (`not`)|Negation|`¬exists(path)`|
|`⊻` (`xor`)|Exclusive disjunction|`a ⊻ b ≝ (a ∨ b) ∧ ¬(a ∧ b)`|
|`⇒` (`implies`)|Implication|`if valid ⇒ score > 0`|
|`⇔` (`iff`)|Equivalence|`x ∈ (A \ B) ⇔ (x ∈ A ∧ x ∉ B)`|
|`⊤` (`true`)|Bool true / lattice top|`found ← ⊤`|
|`⊥` (`false`)|Bool false / lattice bottom|`valid ← ⊥`|
|`None` / `Nil`|Absence / declared alias|`cached ← None`|
|`Result(T,E)`|Tagged success/error|`Ok(T) ⊍ Err(E)`|
|`Option(T)`|Tagged presence/absence|`Some(T) ⊍ {None}`|

```text
Γ ≝ assumptions/typings; ℳ ≝ model
Variant(T) ≝ variant carrying T; Variant(v) ≝ constructed value
Context(⊥) ∈ {Bool, Lattice, Denotation}; Denotation ⇒ LocalLegend
Kinds ≝ Bool ⊍ Absence ⊍ Error ⊍ Divergence
```

`Γ ⊬ φ`: Non-derivability, not refutation or non-satisfaction.

## 3 Evaluation, State & Concurrency

|Symbol|Meaning|Example|
|:---|:---|:---|
|`⇓`|Big-step evaluation|`⟨e,σ⟩ ⇓ ⟨v,σ′⟩`|
|`⇑`|Divergence|`⟨e,σ⟩ ⇑`|
|`⟶`|Small step / state transition|`⟨e,σ⟩ ⟶ ⟨e′,σ′⟩`|
|`↠`|Reflexive-transitive closure of `⟶`|`if s₀ ⟶ s₁ ⟶ s₂ ⇒ s₀ ↠ s₂`|
|`⟵`|Return, not assignment|`⟵ manifest`|
|`⤅`|Yield/suspend|`⤅ sample`|
|`⟦·⟧`|Denotation|`⟦skip⟧(σ) = σ`|
|`⫽`|Lazy absent-value fallback|`v ← cached ⫽ compute()`|
|`⑂`|Spawn; return handle|`t ← ⑂ worker(input)`|
|`⑃`|Await; preserve handle order|`⟨r₁,r₂⟩ ← ⑃ ⟨t₁,t₂⟩`|
|`σ[x ↦ v]`|Store update|`σ′ ← σ[x ↦ v]`|
|`⊕`|Right-biased map override|`σ′ ← σ ⊕ Δσ`|
|`G → A` / `□`|Guarded command / alternative separator|`G₁ → A₁ □ G₂ → A₂`|
|`Δσ`|Finite update map|`Δσ ← {phase ↦ validate}`|
|`⟳`|Repeat named operation|`⟳ Validate`|
|`□`|Temporal always|`□ Inv_scope`|
|`◊`|Temporal eventually|`◊ Terminal(σ)`|

```text
a ⫽ b ≝ let v ← EvalOnce(a) in if v = None then Eval(b) else v
Unwrap(Some(v)) ≝ v; Coalesce ≠ Unwrap
(σ ⊕ Δσ)(k) ≝ if k ∈ Domain(Δσ) then Δσ(k) else σ(k)
Delete ⇒ ExplicitOperation; WorkerFailure ⇒ Error ⊍ Result
GuardedChoice ≝ nondeterministic among enabled alternatives
NoEnabledGuard ⇒ DeclaredRoute(fail|block|terminate)
TemporalFormula ≠ Proof; Retry ≝ ⟨initial, limit, increment, exhaustion⟩
```

### Ordered Rules & Invariant Skeletons

```text
R ≝ ⟨⟨G₁,A₁⟩, …, ⟨Gₙ,Aₙ⟩⟩; positions ≝ 1..n
Gᵢ : State → Bool; Total(Gᵢ) ∧ Pure(Gᵢ)
Enabled(R, σ) ≝ {i ∈ {1, …, n} : Gᵢ(σ) = ⊤}
Select(R, σ) ≝ if Enabled(R, σ) = ∅ then None else R[min Enabled(R, σ)]
Selected ≝ Select(R,σ) ≠ None
Selected ∧ ¬Admissible ⇒ Inquiry; priority preserved
Ready ≝ Selected ∧ Preconditions ∧ Authorized ∧ Approved ∧ PreserveScope
Prepareτ(σ,trigger,preconditions) ≝ if Ready then ⟨Δσ, σ ⊕ Δσ, Postconditions, Vτ⟩ else None
Skeletonτ ≝ ⟨σ,trigger,preconditions⟩
  ⟶ (Prepareτ(σ,trigger,preconditions) ⫽ ⟨σ,Q⟩)
Payload ≝ plan
Postconditions ≝ result predicates
ActualMutation ⇒ Ready
Inquiry ⇒ PreserveWorktree ∧ Suspend
EvaluationFailure ⇒ ExplicitError
Payload ≠ ExecutedChecks
```

## 4 Sets, Lattices & Aggregates

|Symbol|Meaning|Example|
|:---|:---|:---|
|`∈`|Membership|`x ∈ S`|
|`∉`|Non-membership|`s ∉ R`|
|`⊂`|Proper subset|`A ⊂ Ω`|
|`⊆`|Inclusive subset|`A ⊆ Ω`|
|`∪`|Union|`C ← A ∪ B`|
|`∩`|Intersection|`overlap ← A ∩ B`|
|`⊍` / `⊎`|Tagged sum; project aliases|`Ok(T) ⊍ Err(E)`|
|`\`|Set difference|`pending ← all \ done`|
|`▵`|Symmetric difference; local convention|`(A \ B) ∪ (B \ A)`|
|`×`|Cartesian product|`Input ≝ State × Event`|
|`∅`|Empty set|`if S = ∅ : ⟵ None`|
|`\|S\|`|Cardinality / sequence length|`n ← \|S\|`|
|`𝒫(S)`|Power set|`subsets ← 𝒫(S)`|
|`⊔`|Lattice join; least upper bound|`x ⊔ y`|
|`⊓`|Lattice meet; greatest lower bound|`x ⊓ y`|
|`Σ`|Indexed sum|`Σ_{i=0}^{n−1} a[i]`|
|`Π`|Indexed product|`Π_{i=1}^{k} a_i`|
|`min` / `max`|Attained extrema|`min_{x ∈ S} f(x)`; `max(a,b)`|
|`argmin`|All minimizing arguments|`{x ∈ S : ∀ y ∈ S : f(x) ≤ f(y)}`|
|`argmax`|All maximizing arguments|`{x ∈ S : ∀ y ∈ S : f(x) ≥ f(y)}`|
|`{x ∈ S : P(x)}`|Comprehension|`{x ∈ S : score(x) > θ}`|

```text
TaggedSum ⇒ unique variant; overlapping payloads permitted
⊍ ≝ ⊎ locally; global glyph equivalence unclaimed
EmptySum ≝ 0 ∧ EmptyProduct ≝ 1; domain identities required
Ordinary(min/max) ⇒ NonemptyDomain ∧ AttainedExtremum
ExtendedValues ⇒ LocalLegend
ArgExtrema ⊆ S
SelectOne(ArgExtrema) ⇒ Exists ∧ ExplicitTieBreak
```

## 5 Quantifiers & Selection

|Symbol|Meaning|Example|
|:---|:---|:---|
|`∀`|Universal quantifier|`∀ x ∈ S : P(x)`|
|`∃`|Existential quantifier|`∃ x ∈ S : P(x)`|
|`∄`|No satisfying element|`∄ x ∈ S : P(x)`|
|`∃!`|Exactly one satisfying element|`∃! x ∈ S : P(x)`|
|`:`|Predicate scope / typing|`∀ x ∈ S : P(x)`; `v : T`|
|`for x ∈ S : A`|Iteration, not quantification|`for s ∈ items : s.active ← ⊤`|

```text
(∀ x ∈ ∅ : P(x)) = ⊤; (∃ x ∈ ∅ : P(x)) = (∃! x ∈ ∅ : P(x)) = ⊥
OrderSensitiveEffects ⇒ OrderedInput ∨ ExplicitOrder
```

## 6 Sequences, Intervals & Signatures

|Symbol|Meaning|Example|
|:---|:---|:---|
|`⟨x₁,…,xₙ⟩`|Ordered tuple/vector|`p ← ⟨x₀,x₁,x₂⟩`|
|`A[i]`|Index; zero-based default|`first ← A[0]`|
|`A[i..j]`|Inclusive slice|`A[0..2]`|
|`A[i:j]`|Half-open slice|`A[0:3]`|
|`⧺`|Sequence concatenation|`"a" ⧺ "b"`|
|`▷`|Sequential left-to-right dataflow|`stream ▷ transform ▷ sink`|
|`[a,b]`|Closed interval|`score ∈ [0,1]`|
|`(a,b)`|Open interval|`residual ∈ (−1,1)`|
|`[a,b)`|Half-open interval|`time ∈ [start,end)`|
|`∞`|Positive infinity; extended domain|`min_cost ← ∞`|
|`NaN`|Floating-point not-a-number|`if isNaN(v) : ⟵ Err(NaNValue)`|
|`ε`|Positive tolerance|`step_small ← \|x′−x\| ≤ ε`|

```text
Index ⇒ 0 ≤ i < |A|
InclusiveSlice ⇒ 0 ≤ i ≤ j < |A|
HalfOpenSlice ⇒ 0 ≤ i ≤ j ≤ |A|; LocalIndexBase overrides default
TypeArrow ≠ Transition
Partiality/Error/Async ⇒ ExplicitContract
IEEE: NaN = NaN is false; detection ≝ isNaN(v)
ε > 0; ε ≠ infinitesimal
```

## 7 Asymptotic Complexity

|Symbol|Meaning|Usage|
|:---|:---|:---|
|`O(g(n))`|Constant-factor upper bound|`n ∈ O(n²)`|
|`Ω(g(n))`|Constant-factor lower bound|`n² ∈ Ω(n)`|
|`Θ(g(n))`|Both bounds|`3n+2 ∈ Θ(n)`|
|`o(g(n))`|Vanishing relative ratio|`1/n ∈ o(1)`|

```text
n → ∞; f ≥ 0; g eventually > 0
Bounds ≝ function classes
f ∈ o(g) ⇔ f(n)/g(n) → 0
CostClaim ≝ ⟨size measure, computational model, worst/average/expected/other case⟩
```

## 8 Epistemic & Protocol Tags

|Symbol|Category|Semantic Meaning|Example|
|:---|:---|:---|:---|
|`𝒟`|Evidence|Inspected; anchor-bounded|`[𝒟 \| output] exit = 0`|
|`ℐ`|Authority|Governing contract|`[ℐ \| contract] scope`|
|`ℋ`|Hypothesis|Confidence + falsifier|`[ℋ \| conf:low \| falsifier:test] P`|
|`𝒫`|Proposal|Operation + approval gate|`[𝒫 \| action:patch \| approval:req] update`|
|`𝒬`|Inquiry|Answerable uncertainty|`[𝒬 \| topic:runtime] available?`|

```text
Claim ⇒ Atomic ∧ Attributable; Tag ≠ ScientificProof
EvidenceSupports(P) ⇒ NewAnchoredClaim(P); HypothesisTag retained
approval:opt ⇒ ExplicitTaskAuthorization; GateMetadata ≠ GrantedApproval
CheckRecord ≝ ⟨availability, execution, exit, human_review⟩
𝒫(S) ≝ power set; 𝒫 ≝ proposal; indexed Σ/Π ≝ sum/product
```