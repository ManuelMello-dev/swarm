# swarm

## Install

Install the runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

The repository includes `adjudicator`, its bounded JAX memory implementation, and the QPSO/role-update module imported by the adjudicator.

## Learnable quantum-memory projection

The memory ring retains each encoded grid's 73-dimensional feature vector. After feedback, the adjudicator updates the real and imaginary projection matrices every eight outcomes by default. The objective favors memories with higher reward, greater emotional weight, and stronger task relevance. After a successful update, all feature-bearing entries are re-embedded with the new matrices so recall compares embeddings from the same projection version. Configure `quantum_projection_learning_rate` and `quantum_projection_update_every` through `SwarmConfig` to tune update size and cadence.

## Tests

Run the CPU JAX tests with:

```bash
python -m unittest discover -s tests -v
```
