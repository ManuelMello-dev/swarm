import importlib.machinery
import importlib.util
import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
loader = importlib.machinery.SourceFileLoader("adjudicator", str(REPO_ROOT / "adjudicator"))
spec = importlib.util.spec_from_loader("adjudicator", loader)
adjudicator = importlib.util.module_from_spec(spec)
sys.modules["adjudicator"] = adjudicator
loader.exec_module(adjudicator)
JAXProposalSwarm = adjudicator.JAXProposalSwarm
SwarmConfig = adjudicator.SwarmConfig

from jax_quantum_memory import FEATURE_DIM, JAXQuantumMemoryField, project_features


class QuantumMemoryProjectionTests(unittest.TestCase):
    def test_projection_update_reembeds_only_feature_bearing_memories(self):
        memory = JAXQuantumMemoryField(dim=8, capacity=4, top_k=2, seed=17)
        feature_a = jnp.zeros((FEATURE_DIM,), jnp.float32).at[0].set(1.0)
        feature_b = jnp.zeros((FEATURE_DIM,), jnp.float32).at[1].set(1.0)
        amp_a = project_features(feature_a, memory.proj_real, memory.proj_imag)
        amp_b = project_features(feature_b, memory.proj_real, memory.proj_imag)

        memory.state = memory.candidate_encode(
            amp_a, action_id=1, reward=-0.5, task_id=9, phase=0.0,
            emotional_weight=0.5, features=feature_a,
        )
        memory.state = memory.candidate_encode(
            amp_b, action_id=2, reward=1.0, task_id=9, phase=1.0,
            emotional_weight=1.0, features=feature_b,
        )
        memory.state = memory.candidate_encode(
            amp_a, action_id=3, reward=0.0, task_id=9, phase=2.0,
            emotional_weight=1.0,
        )

        old_real = np.asarray(memory.proj_real).copy()
        old_imag = np.asarray(memory.proj_imag).copy()
        old_featureless_amplitude = np.asarray(memory.state.amplitudes[2]).copy()
        loss, grad_norm, updated = memory.update_projection(feature_b, task_id=9, learning_rate=0.01)
        jax.block_until_ready((loss, grad_norm, updated, memory.state.amplitudes))

        self.assertTrue(bool(updated))
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(np.isfinite(float(grad_norm)))
        self.assertGreater(float(grad_norm), 0.0)
        self.assertFalse(np.array_equal(old_real, np.asarray(memory.proj_real)))
        self.assertFalse(np.array_equal(old_imag, np.asarray(memory.proj_imag)))
        np.testing.assert_array_equal(memory.state.amplitudes[2], old_featureless_amplitude)
        self.assertEqual(memory.state.feature_valid.tolist(), [True, True, False, False])

        expected = jax.vmap(
            lambda feature: project_features(feature, memory.proj_real, memory.proj_imag)
        )(memory.state.features[:2])
        np.testing.assert_allclose(memory.state.amplitudes[:2], expected, rtol=1e-5, atol=1e-6)
        norms = np.linalg.norm(np.asarray(memory.state.amplitudes[:2]), axis=-1)
        np.testing.assert_allclose(norms, np.ones(2), rtol=1e-5, atol=1e-6)

    def test_update_skips_until_two_feature_bearing_memories_exist(self):
        memory = JAXQuantumMemoryField(dim=8, capacity=4, top_k=2, seed=23)
        feature = jnp.zeros((FEATURE_DIM,), jnp.float32).at[3].set(1.0)
        amplitudes = memory.encode_grid(jnp.zeros((2, 2), jnp.int32))
        memory.state = memory.candidate_encode(
            amplitudes, action_id=1, reward=1.0, task_id=1, phase=0.0,
            features=feature,
        )
        memory.state = memory.candidate_encode(
            amplitudes, action_id=2, reward=0.0, task_id=1, phase=1.0,
        )
        before_real = np.asarray(memory.proj_real).copy()
        _, _, updated = memory.update_projection(feature, task_id=1, learning_rate=0.01)
        self.assertFalse(bool(updated))
        np.testing.assert_array_equal(memory.proj_real, before_real)

    def test_adjudicator_carries_features_through_feedback_and_updates_on_cadence(self):
        config = SwarmConfig(
            max_agents=2,
            min_agents=2,
            chunk_size=2,
            grid_side=4,
            hidden_dim=8,
            graph_capacity=8,
            retrieval_top_k=2,
            quantum_memory_dim=8,
            quantum_memory_capacity=8,
            quantum_memory_top_k=2,
            quantum_projection_learning_rate=0.01,
            quantum_projection_update_every=1,
            num_roles=2,
            qpso_particles=2,
            qpso_every=100,
            rotation_every=100,
            seed=5,
        )
        swarm = JAXProposalSwarm(config)
        grids = [np.zeros((4, 4), np.int32), np.eye(4, dtype=np.int32)]
        outcomes = []
        for index, grid in enumerate(grids):
            swarm.propose(grid, legal_actions=[1, 2, 6], task_id=4)
            outcomes.append(swarm.observe_outcome(reward=float(index), task_id=4))

        self.assertFalse(outcomes[0]["projection_updated"])
        self.assertTrue(outcomes[1]["projection_updated"])
        self.assertEqual(int(swarm.quantum_memory.state.count), 2)
        self.assertTrue(bool(jnp.all(swarm.quantum_memory.state.feature_valid[:2])))
        self.assertGreater(outcomes[1]["projection_grad_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
