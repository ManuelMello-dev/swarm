import importlib.machinery
import importlib.util
import sys
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
if "adjudicator" not in sys.modules:
    loader = importlib.machinery.SourceFileLoader("adjudicator", str(REPO_ROOT / "adjudicator"))
    spec = importlib.util.spec_from_loader("adjudicator", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["adjudicator"] = module
    loader.exec_module(module)
adjudicator = sys.modules["adjudicator"]


def config(**overrides):
    values = dict(
        max_agents=4, min_agents=3, chunk_size=2, grid_side=4, hidden_dim=8,
        graph_capacity=8, retrieval_top_k=2, quantum_memory_dim=8,
        quantum_memory_capacity=8, quantum_memory_top_k=2, num_roles=2,
        qpso_particles=2, qpso_every=100, rotation_every=100, seed=13,
    )
    values.update(overrides)
    return adjudicator.SwarmConfig(**values)


class TrunkLearningTests(unittest.TestCase):
    def test_recomputed_trunk_matches_proposal_hiddens(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        rng = np.random.default_rng(1)
        for _ in range(2):  # second round has non-zero recurrent input
            swarm.propose(rng.integers(0, 10, (5, 6)), legal_actions=[1, 2, 6], task_id=1)
            proposals = swarm._pending.proposals
            recomputed = adjudicator._recompute_trunk(swarm.params, proposals)
            mask = np.asarray(proposals.active_mask)
            np.testing.assert_allclose(
                np.asarray(recomputed)[mask], np.asarray(proposals.agent_hidden)[mask],
                rtol=1e-5, atol=1e-6,
            )
            swarm.observe_outcome(reward=0.5, next_grid=rng.integers(0, 10, (5, 6)), task_id=1)

    def test_every_trunk_weight_receives_updates(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        before = {name: np.asarray(getattr(swarm.params, name)).copy()
                  for name in adjudicator.TRUNK_FIELDS}
        rng = np.random.default_rng(2)
        for _ in range(2):
            swarm.propose(rng.integers(0, 10, (4, 4)), legal_actions=[1, 2, 6], task_id=1)
            feedback = swarm.observe_outcome(
                reward=1.0, next_grid=rng.integers(0, 10, (4, 4)), task_id=1
            )
            self.assertGreater(feedback["trunk_grad_norm"], 0.0)
            self.assertTrue(np.isfinite(feedback["trunk_loss"]))
        for name, old in before.items():
            self.assertFalse(np.array_equal(old, np.asarray(getattr(swarm.params, name))), name)

    def test_trunk_step_reduces_its_loss(self):
        swarm = adjudicator.JAXProposalSwarm(config(
            trunk_learning_rate=0.05, value_learning_rate=0.0, prediction_learning_rate=0.0,
            policy_learning_rate=0.0, coordinate_learning_rate=0.0,
        ))
        rng = np.random.default_rng(3)
        grid = rng.integers(0, 10, (4, 4))
        swarm.propose(grid, legal_actions=[1, 2], task_id=1)
        pending, state = swarm._pending, swarm.state
        args = (jnp.asarray(1.0, jnp.float32),
                jnp.asarray(rng.integers(0, 10, 16), jnp.int32), jnp.asarray(True))
        params = swarm.params
        losses = []
        for _ in range(5):
            params, _, _, _, diagnostics = swarm._feedback_jit(params, state, pending, *args)
            losses.append(float(diagnostics[2]))
        self.assertLess(losses[-1], losses[0])
        swarm.cancel_pending()

    def test_zero_trunk_rate_freezes_trunk(self):
        swarm = adjudicator.JAXProposalSwarm(config(trunk_learning_rate=0.0))
        before = np.asarray(swarm.params.comm_query).copy()
        swarm.propose(np.ones((4, 4), np.int32), legal_actions=[1, 2], task_id=1)
        swarm.observe_outcome(reward=1.0, next_grid=np.zeros((4, 4), np.int32), task_id=1)
        np.testing.assert_array_equal(np.asarray(swarm.params.comm_query), before)


if __name__ == "__main__":
    unittest.main()
