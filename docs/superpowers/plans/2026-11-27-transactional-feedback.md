# Transactional Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent partial feedback commits and duplicate learning while preserving the newer relative-surprise weighting behavior.

**Architecture:** Stage all feedback, memory/projection, surprise-history, role and QPSO updates in immutable snapshots, synchronize and validate candidates, then replace the runtime bundle once under a shared reentrant lock. Give each proposal a stable ID and atomically retain its canonical outcome and cached result. Keep execution receipts outside learning retries so failed feedback never repeats an action.

**Tech Stack:** Python, JAX, NumPy, unittest, GitHub CLI.

**Spec:** This document implements the user-approved transaction and retry design from this task. Base revision: `49c78221f6db9e530a2cf2cdfc4882c8b8eb5be2`.

## Global Constraints

Preserve the existing positional `observe_outcome(reward, next_grid=None, task_id=None)` API; add an optional keyword-only proposal ID. Explicit IDs are required for duplicate delivery after completion.

Do not change learning formulas, learning order, default cadence, or relative emotional weighting. Include surprise mean/variance/count in the transaction.

Use no new runtime dependency. All host mutations, including direct memory APIs, share an RLock. This protects same-process retries and concurrency, not crash/restart recovery; document that boundary.

Push one tested commit to existing `main` without force; do not overwrite concurrent upstream changes.

## Files and interfaces

`jax_quantum_memory.py` owns `MemorySnapshot(state, proj_real, proj_imag)`, standalone snapshot-backed compatibility properties, snapshot binding to a swarm owner with the same lock, and `candidate_projection(snapshot, query_features, task_id, learning_rate)`. It returns `(next_snapshot, loss, grad_norm, updated)` without modifying any owner. Existing `update_projection` remains supported as a synchronized explicit mutation.

`adjudicator` owns `RuntimeBundle`, stable proposal IDs on `Decision`, outcome fingerprints and completion records, pure feedback-candidate computation, and the single commit. Existing writable parameter/state access remains available through lock-protected compatibility properties. `step` stores an execution receipt before parsing and calls `retry_feedback` without another execution. An exception from execution is an unknown outcome and requires explicit reconciliation.

`tests/test_quantum_memory_candidates.py` covers candidate purity, explicit input snapshots, binding and legacy behavior. `tests/test_feedback_transactions.py` covers failure injection, deterministic retry, same-ID duplicates, conflicting outcomes, unknown/mismatched IDs, distinct identical outcomes, diagnostic/synchronization failure, external execution/parse failures, and concurrent duplicate feedback.

`README.md` documents the retry API, explicit-ID deduplication and process-local limits.

## Task 1: Nonmutating memory/projection candidates

- [x] Write candidate-purity and explicit-snapshot tests, run to confirm missing-API failures.
- [x] Add the immutable memory snapshot and lock-backed compatibility API. Candidate projection calls the existing JIT training function with the supplied snapshot, not live state.
- [x] Run projection tests, including original refresh/eligibility tests.

Concrete API:

```python
candidate, loss, grad_norm, updated = memory.candidate_projection(
    input_snapshot, query_features, task_id=4, learning_rate=0.01)
assert memory.snapshot is original_snapshot
```

## Task 2: Atomic feedback and stable IDs

- [x] Add failure/duplicate tests before changing the feedback implementation.
- [x] Build one runtime bundle containing all learning and control state; compute feedback candidates from one captured bundle with no owner mutation.
- [x] Synchronize the candidate JAX pytree, check finite floating/complex leaves and diagnostics, and build the Python result before committing. Publish the learning snapshot, completion record and cleared pending state in one pointer replacement under the shared lock.
- [x] Preserve the pre-feedback baseline and current insertion-before-projection order. Relative emotional-weight history is computed as a candidate rather than mutated early.
- [x] Verify retries after injected projection, role, QPSO, synchronization and diagnostic failures match a clean single update.

Concrete retry API:

```python
decision = swarm.propose(grid, [1, 2, 6], task_id=4)
feedback = swarm.observe_outcome(1.0, next_grid, task_id=4,
                                 proposal_id=decision.proposal_id)
assert swarm.observe_outcome(1.0, next_grid, task_id=4,
                             proposal_id=decision.proposal_id) == feedback
```

## Task 3: Action/feedback boundary and delivery

- [x] Store execution receipt and parsed outcome on the pending proposal; retain them on parsing or feedback failure. `retry_feedback(parse_outcome=None)` parses a saved receipt when necessary and retries learning only.
- [x] Mark execution exceptions as uncertain, block implicit repetition/cancellation, and permit explicit outcome reconciliation through `observe_outcome`.
- [x] Document retry and process-local deduplication semantics; add full-loop and concurrency tests.
- [x] Run `python -m unittest discover -s tests -v`, independent code review, and `git diff --check`.
- [ ] Commit all relevant source, tests and documentation, push without force, and verify remote `main` equals the new commit.

## Verification before commit

The full CPU suite passes: 43 tests in 109.308 seconds, JAX 0.12.1 and NumPy 2.5.4. Regression tests reproduced partial commits before the refactor and now cover deterministic retries, projection/role/QPSO/synchronization/diagnostic failures, nonfinite candidates, concurrent duplicates, receipt parsing, uncertain execution and episode-reset deduplication. All 19 existing top-level adjudicator functions and nine memory functions are AST-identical to the base revision; the numerical learning kernels were not changed. Independent review identified a cadence-reference error, which was corrected before this successful full run. Callback lock constraints and process-local completion storage are documented.
