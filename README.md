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
## Tests

Run the CPU JAX tests with:

```bash
python -m unittest discover -s tests -v
```
