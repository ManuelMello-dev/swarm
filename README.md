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

## Tests

Run the CPU JAX tests with:

```bash
python -m unittest discover -s tests -v
```
