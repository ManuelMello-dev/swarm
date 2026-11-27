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


def small_config(**overrides):
    values = dict(
        max_agents=2, min_agents=2, chunk_size=2, grid_side=4, hidden_dim=8,
        graph_capacity=8, retrieval_top_k=2, quantum_memory_dim=8,
        quantum_memory_capacity=8, quantum_memory_top_k=2, num_roles=2,
        qpso_particles=2, qpso_every=100, rotation_every=100, seed=11,
    )
    values.update(overrides)
    return adjudicator.SwarmConfig(**values)


def propose_args(swarm, grid, task_id=1):
    return (
        jnp.asarray(grid, jnp.int32), jnp.ones((6,), bool), jnp.asarray(task_id, jnp.int32),
        jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32),
        jnp.asarray(swarm.last_memory_features, jnp.float32), jnp.asarray(2, jnp.int32),
    )


class ComponentConnectionTests(unittest.TestCase):
    def test_qpso_baseline_parameters_reach_policy_update(self):
        swarm = adjudicator.JAXProposalSwarm(small_config())
        swarm.propose(np.ones((4, 4), np.int32), legal_actions=[1, 2], task_id=1)
        pending, params, state = swarm._pending, swarm.params, swarm.state
        reward = jnp.asarray(1.0, jnp.float32)
        colors = jnp.zeros((16,), jnp.int32)
        has_next = jnp.asarray(True)
        base, *_ = swarm._feedback_jit(params, state, pending, reward, colors, has_next)
        shifted = params._replace(baseline_bias=params.baseline_bias + 0.5)
        moved, _, _, surprise, _ = swarm._feedback_jit(shifted, state, pending, reward, colors, has_next)
        self.assertFalse(np.allclose(base.agent_action_bias, moved.agent_action_bias))
        self.assertTrue(np.isfinite(float(surprise)))
        swarm.cancel_pending()

    def test_macrograph_outcomes_bias_action_values(self):
        swarm = adjudicator.JAXProposalSwarm(small_config())
        grid = np.ones((4, 4), np.int32)
        swarm.propose(grid, legal_actions=[1, 2], task_id=1)
        swarm.observe_outcome(reward=1.0, next_grid=grid, task_id=1)
        graph = swarm.state.graph
        self.assertEqual(int(graph.count), 1)

        args = propose_args(swarm, grid)
        good = swarm._propose_jit(swarm.params, swarm.state, *args)
        bad_state = swarm.state._replace(graph=graph._replace(rewards=-graph.rewards))
        bad = swarm._propose_jit(swarm.params, bad_state, *args)

        chosen = int(graph.actions[0])
        self.assertGreater(float(good.action_prior[chosen]), 0.0)
        self.assertLess(float(bad.action_prior[chosen]), 0.0)
        unvisited = [a for a in range(6) if a != chosen]
        np.testing.assert_array_equal(np.asarray(good.action_prior)[unvisited], 0.0)
        self.assertGreater(
            float(good.action_values[0, chosen]), float(bad.action_values[0, chosen])
        )

    def test_action_prior_is_zero_without_memory_or_when_disabled(self):
        grid = np.ones((4, 4), np.int32)
        swarm = adjudicator.JAXProposalSwarm(small_config())
        empty = swarm._propose_jit(swarm.params, swarm.state, *propose_args(swarm, grid))
        np.testing.assert_array_equal(np.asarray(empty.action_prior), 0.0)

        disabled = adjudicator.JAXProposalSwarm(small_config(graph_action_prior_weight=0.0))
        disabled.propose(grid, legal_actions=[1, 2], task_id=1)
        disabled.observe_outcome(reward=1.0, next_grid=grid, task_id=1)
        out = disabled._propose_jit(
            disabled.params, disabled.state, *propose_args(disabled, grid)
        )
        np.testing.assert_array_equal(np.asarray(out.action_prior), 0.0)

    def test_full_loop_with_qpso_and_rotation_stays_finite(self):
        swarm = adjudicator.JAXProposalSwarm(small_config(qpso_every=2, rotation_every=2))
        rng = np.random.default_rng(0)
        for _ in range(8):
            grid = rng.integers(0, 10, (4, 4))
            decision = swarm.propose(grid, legal_actions=[1, 2, 6], task_id=3)
            feedback = swarm.observe_outcome(
                reward=float(decision.action_id == 2), next_grid=grid, task_id=3
            )
        self.assertEqual(feedback["qpso_generation"], 4)
        for key in ("mean_reliability", "reward_baseline", "surprise", "qpso_best_fitness"):
            self.assertTrue(np.isfinite(feedback[key]), key)



class CoordinateAndQPSOTests(unittest.TestCase):
    def _coords(self, swarm, params, state, grid):
        out = swarm._propose_jit(params, state, *propose_args(swarm, grid))
        return np.asarray(out.coordinates[:2])

    def test_click_coordinates_move_toward_observed_change(self):
        swarm = adjudicator.JAXProposalSwarm(small_config(coordinate_learning_rate=0.5))
        grid = np.zeros((4, 4), np.int32)
        changed = grid.copy()
        changed[3, 3] = 5  # bottom-right cell changes -> target (1.0, 1.0)
        target = np.array([1.0, 1.0])
        initial_params = swarm.params
        for _ in range(6):
            decision = swarm.propose(grid, legal_actions=[6], task_id=2)
            self.assertEqual(decision.action_id, 6)
            feedback = swarm.observe_outcome(reward=0.0, next_grid=changed, task_id=2)
            self.assertEqual(feedback["coordinate_update_weight"], 1.0)
        before = self._coords(swarm, initial_params, swarm.state, grid)
        after = self._coords(swarm, swarm.params, swarm.state, grid)
        self.assertLess(
            np.linalg.norm(after - target, axis=-1).mean(),
            np.linalg.norm(before - target, axis=-1).mean(),
        )

    def test_coordinates_untouched_when_not_clicking(self):
        swarm = adjudicator.JAXProposalSwarm(small_config())
        grid = np.zeros((4, 4), np.int32)
        before = np.asarray(swarm.params.coordinate_projection).copy()
        swarm.propose(grid, legal_actions=[1], task_id=2)
        feedback = swarm.observe_outcome(reward=1.0, next_grid=grid + 1, task_id=2)
        self.assertEqual(feedback["coordinate_update_weight"], 0.0)
        np.testing.assert_array_equal(np.asarray(swarm.params.coordinate_projection), before)

    def test_qpso_uses_refreshed_best_fitness(self):
        import jax
        from qPSO_and_role_updates import init_qpso, qpso_step
        state = init_qpso(jax.random.PRNGKey(0), param_dim=3, n_particles=2)
        stale = state._replace(
            particles=state.particles._replace(personal_best_fitness=jnp.array([100.0, 100.0])),
            global_best_fitness=jnp.asarray(100.0),
        )
        fitness = jnp.array([1.0, 2.0])
        key = jax.random.PRNGKey(1)
        frozen = qpso_step(stale, fitness, key)
        np.testing.assert_array_equal(frozen.particles.personal_best_fitness, [100.0, 100.0])
        refreshed = qpso_step(
            stale, fitness, key,
            personal_best_fitness=jnp.array([0.0, 0.0]),
            global_best_fitness=jnp.asarray(0.0),
        )
        np.testing.assert_array_equal(refreshed.particles.personal_best_fitness, [1.0, 2.0])
        self.assertEqual(float(refreshed.global_best_fitness), 2.0)
        np.testing.assert_array_equal(refreshed.global_best, stale.particles.position[1])


if __name__ == "__main__":
    unittest.main()
