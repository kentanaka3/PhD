# Positive Contracts & Invariant Skeletons

## 1. Operational Transformation

Use the [canonical notation](symbolism.md). Optimize symbolic density and
generality subject to explicit semantics and preserved authority.

```text
T : Constraint → ⟨AdmissibleSet, Preservation, OrderedRules, Inquiry⟩
Equivalent(C, T(C)) ≝ ∀ σ, a : Allowed_C(σ, a) ⇔ Allowed_T(C)(σ, a)
Wτ ≝ ExplicitTargets ∩ PermittedPaths ∩ AuthorizedPaths
Aτ ≝ AuthorizedActions
Vτ ≝ RequiredChecks
```

`Equivalent` is a human-review obligation; schema checks assess structure.
Derive admissible sets from the original contract. A narrower set requires
explicit agreement just as a broader set does.

| Boundary intent | Positive canonical template |
|:---|:---|
| Preserve unrelated paths | `∀ p ∈ (Domain(σ) ∪ Domain(σ′)) \ Wτ : σ′(p) ≡ σ(p)` |
| Preserve unselected symbols | `∀ n ∈ UnselectedNodes : AST′(n) ≡ AST(n)` |
| Preserve existing target edits | `PatchGate ≝ Inspected ∧ IntegratedUserChanges ∧ Aligned ∧ Approved` |
| Bound tool/data access | `action ∈ Aτ ∧ source ∈ AuthorizedSourcesτ` |
| Gate consequential actions | `Execute(a) ⇒ Preconditions(a) ∧ ExplicitApproval(a)` |
| Resolve uncertainty | `MaterialQuestion ∨ Conflict ⇒ InquireAndAlign` |

Logical negation and set difference remain available for precise predicates.
The transformation preserves behavior rather than enforcing a lexical ban.

## 2. Priority Dispatch

```text
R ≝ ⟨⟨G₁, A₁⟩, …, ⟨Gₙ, Aₙ⟩⟩     [highest priority first]
Match(R, σ, e) ≝ first r ∈ R with r.guard(σ, e) = ⊤ ⫽ None
Dispatch(σ, e) ≝
  r ← Match(R, σ, e)
  r = None ⇒ ⟵ ⟨σ, InquireAndAlign⟩
  Admissible(r.action, σ) ⇒ ⟵ Execute(r.action, σ)
  otherwise ⇒ ⟵ ⟨σ, InquireAndAlign⟩
```

Place authority, conflict, and material-uncertainty guards before ordinary
workflow guards. Inquiry emits one focused question, preserves state, and
suspends execution until answered. Finite retry bounds and terminal states
are explicit; fallback denotes inquiry, not proof of eventual completion.

## 3. Parameterized Invariant Skeleton

```text
Skeletonτ ≝ ⟨σ, trigger, preconditions⟩
  ⟶ ⟨Δσ, σ′, postconditions, Vτ⟩
  where action ∈ Aτ ∧ Effects(action, σ) ⊆ Wτ
        ∧ σ′ = Apply(σ, Δσ) ∧ Preserve(σ, σ′)
  ⫽ ⟨σ, InquireAndAlign⟩
```

Pre-calculate structure and obligations; bind methods/inputs at task time.
Reference skeletons are specifications; instantiated JSON/YAML examples are
machine-checkable fixtures only when an executable checker is supplied.
Entry points carry a minimal local legend; specialized notation loads on demand.

## 4. Mechanisms & Evidence Boundary

Autoregressive generation conditions successive tokens on prior context.
Attention scores and output logits are distinct quantities; mention of an
action alone establishes neither a positive output bias nor its execution.
Model-specific pretraining distributions are unknown here.

| Claim | Evidence status | Discriminating evaluation |
|:---|:---|:---|
| Positive templates improve compliance over equivalent prohibitions | Hypothesis; unverified here | Matched prompts/tasks; compare boundary adherence and completion |
| Ordered dispatch and skeletons reduce ambiguity | Design rationale; model benefit unverified | Compare conflicting-guard and unmapped-input outcomes |
| Symbolic compression reduces tokens or token competition | Hypothesis; tokenizer/model dependent | Count actual tokens; separately compare interpretation/compliance |
| Canonical notation aligns with pretraining distributions | Hypothesis; distribution unknown | Compare notation variants across identified model versions |

Mechanism statements above are background explanation, not a cited research
review. No comparative evaluation or causal attention analysis accompanies
this specification. Any future result records model/version, tokenizer,
prompt variants, tasks, settings, measurements, and human review.

## 5. Acceptance

```text
AuthoringReview ≝ DefinedSymbols ∧ PreservedBoundaries ∧ ExplicitPriority
                  ∧ InquiryFallback ∧ TaskParameters ∧ EvidenceSeparated
Acceptance ≝ SchemaPassed ∧ HumanReviewed
```

Record skipped/unavailable validation separately from pass. Human review
assesses semantic equivalence; acceptance establishes neither measured
performance gains nor a causal attention/pretraining benefit.
