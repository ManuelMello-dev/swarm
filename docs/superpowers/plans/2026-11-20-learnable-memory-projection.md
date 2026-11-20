# Learnable Memory Projection Implementation Plan

**Goal:** Preserve raw grid features in quantum memory, optimize the shared complex projection from outcome-weighted retrieval targets, and keep stored amplitudes consistent after projection changes.

**Architecture:** Add feature vectors and a feature-valid mask to the bounded memory ring. On configurable feedback intervals, optimize the shared real/imaginary projection with a JAX cross-entropy objective over valid feature-bearing memories, weighted by reward, emotional weight, and task relevance. Re-embed valid entries after each update. `adjudicator` supplies pending raw features and triggers updates after feedback.

**Tech Stack:** Python, JAX, NumPy, standard-library `unittest`.

**Spec:** User request for feature storage, JAX projection learning, re-embedding, performance review, and a commit to `main`.

## Completed changes

- Added `jax_quantum_memory.py` and the imported `qPSO_and_role_updates.py` module to the repository, plus a runtime `requirements.txt`.
- Preserved each memory's raw 73-feature vector and marked featureless entries so they are excluded from training.
- Added an outcome-weighted cross-entropy update over normalized complex overlaps, clipping the joint gradient norm before updating both projection matrices.
- Re-embedded feature-bearing memories after each successful update; the adjudicator defaults to one scheduled update every eight outcomes.
- Added feedback diagnostics and documentation.
- Added tests for feature retention, finite gradients, matrix updates, valid-only re-embedding, featureless-entry exclusion, and the adjudicator feedback lifecycle.

## Verification and delivery

- Installed declared runtime dependencies in an isolated test environment.
- `python -m unittest discover -s tests -v`: 3 tests passed.
- Compiled the extensionless `adjudicator`, both JAX modules, and the test module successfully.
- Ran `git diff --check` successfully and committed the changes to `main`.
