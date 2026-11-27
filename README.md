# swarm

## Install

Install the runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

The repository includes `adjudicator`, its bounded JAX memory implementation, and the QPSO/role-update module imported by the adjudicator.

## Learnable quantum-memory projection

The memory ring retains each encoded grid's 73-dimensional feature vector. After feedback, the adjudicator updates the real and imaginary projection matrices every eight outcomes by default. The objective favors memories with higher reward, greater emotional weight, and stronger task relevance. After a successful update, all feature-bearing entries are re-embedded with the new matrices so recall compares embeddings from the same projection version. Configure `quantum_projection_learning_rate` and `quantum_projection_update_every` through `SwarmConfig` to tune update size and cadence.

## Component connections
The macrograph's stored action/reward outcomes feed back into decisions: retrieval computes the reward observed for each action in the retrieved neighborhood and adds it, scaled by retrieval score and `graph_action_prior_weight` (default 0.5, set 0 to disable), to every agent's action values. The value regression trains on the residual above this prior.

The QPSO-evolved `baseline_weights`/`baseline_bias` form a state-conditioned residual on top of the reward EMA. The policy advantage and the surprise signal (which sets emotional weight in quantum memory) use `EMA + residual`, and QPSO fitness scores particles by how well that combined baseline predicts the observed reward.
ACTION6 click coordinates now learn. After a click, if the next grid is observed and cells changed, every active agent's coordinate head is regressed toward the centroid of the changed cells. If nothing visibly changed, the executed click is reinforced in proportion to positive advantage only. Non-click actions leave the coordinate head untouched. Tune with `coordinate_learning_rate` (default 0.05); feedback reports `coordinate_loss` and `coordinate_update_weight`.

QPSO is scored on a new sample each generation, so stored personal and global bests are re-scored on the current sample before comparison (`qpso_step(..., personal_best_fitness=, global_best_fitness=)`), preventing an early lucky score from locking the global best.

The agent trunk now learns. Recurrence (`recurrent_projection`), token attention (`query_projection`, `key_projection`, `evidence_projection`), and agent-to-agent communication (`comm_query`, `comm_key`, `comm_value`, `comm_gate`) are trained by backpropagating the value, forecast (when the next grid is observed), and click losses through a recomputation of the trunk at feedback time. The proposal keeps the trunk inputs it used, and proposal and feedback call the same trunk function, so the recomputed hiddens match the proposed ones exactly. The update uses `trunk_learning_rate` (default 0.002; 0 freezes the trunk) with a joint gradient-norm clip `trunk_grad_clip_norm` (default 1.0). Feedback reports `trunk_loss` and `trunk_grad_norm`. Measured cost on CPU with the default config: about 35 ms per step versus 32 ms without trunk learning.

Surprise, which sets each memory's emotional weight, is `|reward - baseline| + prediction_error / ln(10)`, clipped to [0, 1]. Dividing the colour cross-entropy by ln 10 (its chance-level value for 10 colours) keeps prediction error from saturating surprise on its own, so emotional weights spread as forecasting improves.

Emotional weight is relative by default (`relative_emotional_weight=True`): each step's surprise is scored against the running mean and standard deviation of recent surprise (`surprise_ema_decay`, default 0.95), passed through a sigmoid, and mapped to `[emotional_floor, 1]` (default floor 0.05). A memory is weighted for being more surprising than usual, not for being stored early in learning. Set `relative_emotional_weight=False` for the previous absolute rule `0.15 + 0.85 * surprise`.

## Atomic feedback and safe retries

`observe_outcome` stages parameter learning, macrograph/memory insertion, projection training and re-embedding, relative-surprise history, role rotation, and QPSO updates without changing live learning state. It synchronizes JAX work, checks the candidate for nonfinite values, and builds diagnostics before publishing one runtime bundle. A failure leaves the pending proposal available and all learning state unchanged, so retrying applies the outcome once. Public swarm mutations and memory operations share a reentrant lock.

Every `Decision` has a stable `proposal_id`. The existing positional feedback API still works for the currently pending proposal. Pass the ID explicitly when delivering outcomes that might be retried after a successful commit:

```python
decision = swarm.propose(grid, legal_actions=[1, 2, 6], task_id=4)
feedback = swarm.observe_outcome(
    reward, next_grid, task_id=4, proposal_id=decision.proposal_id,
)
# The same ID and canonical outcome return the cached result, without learning.
replayed = swarm.observe_outcome(
    reward, next_grid, task_id=4, proposal_id=decision.proposal_id,
)
```

Conflicting reward, task, or validated next-grid data for a completed ID is rejected. Unknown, cancelled, and mismatched IDs are rejected. Distinct proposals may legitimately have identical outcomes. Completion records survive episode resets and are retained for the lifetime of the swarm object; they are process-local and consume memory as the number of completed proposals grows. Restart-safe deduplication requires durable storage of the learning checkpoint and completion records together, which this repository does not implement. Callers using the legacy API without an ID cannot deduplicate a late delivery against an already completed proposal.

`step` retains the action's raw receipt before parsing. If parsing or learning raises, it keeps the proposal and receipt instead of treating the action as undone. Call `swarm.retry_feedback()` to retry parsing/learning from that receipt, **never to execute the action again**. If the original parser failed, supply a corrected one with `swarm.retry_feedback(parse_outcome=parser)`. This method returns the same `(decision, raw_outcome, feedback)` shape as `step`. Once an outcome has parsed successfully it is frozen for subsequent retries; conflicting replacement outcomes are rejected.

An exception from `execute_action` is an uncertain external outcome, not proof that nothing happened. Reconcile it using `observe_outcome` with the pending decision's ID and the actual outcome. A pending executed/uncertain action blocks another proposal and ordinary `cancel_pending()`. Deliberate abandonment is available through `cancel_pending(abandon_executed=True)`; it drops pending learning and restores pre-proposal swarm state, but cannot undo an external action. Do not retry the entire `step` after execution may have happened. The lock serializes calls within this object; it cannot make an external system's actions transactional.

Execution and parser callbacks run while this object's reentrant lock is held. They must not wait for another thread that needs to call into the same swarm. Parsers should only parse the supplied receipt, not mutate or submit feedback to the swarm. Use separate external I/O workers that do not depend on this object's lock if the action callback waits on background work.

## Tests

Run the CPU JAX tests with:

```bash
python -m unittest discover -s tests -v
```
