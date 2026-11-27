import dataclasses
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax_quantum_memory as quantum_memory
from jax_quantum_memory import (
    FEATURE_DIM,
    JAXQuantumMemoryField,
    create_memory_state,
    project_features,
)


class TrackingRLock:
    """A real reentrant lock that records whether operations hold it."""

    def __init__(self):
        self.lock = threading.RLock()
        self.depth = 0
        self.entries = 0

    def __enter__(self):
        self.lock.acquire()
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, *args):
        self.depth -= 1
        self.lock.release()


@dataclasses.dataclass(frozen=True)
class OwnerRuntime:
    memory: object
    marker: int = 0


class QuantumMemoryCandidateTests(unittest.TestCase):
    def setUp(self):
        self.memory = JAXQuantumMemoryField(dim=8, capacity=4, top_k=2, seed=17)
        self.feature_a = jnp.zeros((FEATURE_DIM,), jnp.float32).at[0].set(1.0)
        self.feature_b = jnp.zeros((FEATURE_DIM,), jnp.float32).at[1].set(1.0)

    def assertTreeEqual(self, actual, expected):
        actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
        expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
        self.assertEqual(actual_structure, expected_structure)
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
            np.testing.assert_array_equal(actual_leaf, expected_leaf)

    def populate(self, memory=None):
        memory = self.memory if memory is None else memory
        for index, feature in enumerate((self.feature_a, self.feature_b)):
            amplitudes = project_features(feature, memory.proj_real, memory.proj_imag)
            memory.encode(
                amplitudes, action_id=index + 1, reward=float(index), task_id=9,
                phase=float(index), emotional_weight=0.5 + index * 0.5,
                features=feature,
            )
        amplitudes = project_features(self.feature_a, memory.proj_real, memory.proj_imag)
        memory.encode(amplitudes, action_id=3, reward=0.0, task_id=9, phase=2.0)
        return memory.snapshot

    def test_snapshot_is_immutable_and_compatibility_properties_replace_it(self):
        memory = self.memory
        initial = memory.snapshot
        self.assertIsInstance(initial, quantum_memory.MemorySnapshot)
        self.assertIs(memory._snapshot, initial)
        with self.assertRaises(AttributeError):
            initial.state = create_memory_state(4, 8)

        state = initial.state._replace(total_recalled=jnp.array(5, jnp.int32))
        memory.state = state
        after_state = memory.snapshot
        self.assertIsNot(after_state, initial)
        self.assertIs(after_state.state, state)
        self.assertIs(after_state.proj_real, initial.proj_real)
        self.assertIs(after_state.proj_imag, initial.proj_imag)
        real = initial.proj_real + 0.2
        memory.proj_real = real
        imag = initial.proj_imag - 0.1
        memory.proj_imag = imag
        self.assertIs(memory.snapshot.state, state)
        self.assertIs(memory.snapshot.proj_real, real)
        self.assertIs(memory.snapshot.proj_imag, imag)
        memory.snapshot = initial
        self.assertIs(memory.snapshot, initial)

    def test_candidate_projection_is_pure_and_reembeds_only_eligible_slots(self):
        memory = self.memory
        initial = self.populate()
        before = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), initial)
        candidate, loss, grad_norm, updated = memory.candidate_projection(
            initial, self.feature_b, task_id=9, learning_rate=0.01,
        )
        jax.block_until_ready((candidate, loss, grad_norm, updated))
        self.assertIs(memory.snapshot, initial)
        self.assertTreeEqual(initial, before)
        self.assertTrue(bool(updated))
        self.assertGreater(float(grad_norm), 0.0)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertFalse(np.array_equal(initial.proj_real, candidate.proj_real))
        self.assertFalse(np.array_equal(initial.proj_imag, candidate.proj_imag))
        expected = jax.vmap(
            lambda feature: project_features(feature, candidate.proj_real, candidate.proj_imag)
        )(initial.state.features[:2])
        np.testing.assert_allclose(candidate.state.amplitudes[:2], expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(candidate.state.amplitudes[2:], initial.state.amplitudes[2:])
        for field in initial.state._fields:
            if field != "amplitudes":
                np.testing.assert_array_equal(getattr(candidate.state, field), getattr(initial.state, field))

    def test_candidate_projection_uses_explicit_state_and_both_matrices_not_live(self):
        source = JAXQuantumMemoryField(dim=8, capacity=4, top_k=2, seed=91)
        explicit = self.populate(source)
        live = self.memory.snapshot
        expected_real, expected_imag, expected_state, loss, norm, updated = self.memory._projection_train(
            explicit.proj_real, explicit.proj_imag, explicit.state, self.feature_b,
            jnp.asarray(9, jnp.int32), jnp.asarray(0.01, jnp.float32),
        )
        candidate, actual_loss, actual_norm, actual_updated = self.memory.candidate_projection(
            explicit, self.feature_b, task_id=9, learning_rate=0.01,
        )
        self.assertTreeEqual(candidate, quantum_memory.MemorySnapshot(expected_state, expected_real, expected_imag))
        self.assertTreeEqual((actual_loss, actual_norm, actual_updated), (loss, norm, updated))
        self.assertTrue(bool(actual_updated))
        self.assertIs(self.memory.snapshot, live)
        self.assertEqual(int(self.memory.state.count), 0)

        # Reversing eligibility proves the supplied state controls the skip too.
        candidate, loss, norm, updated = source.candidate_projection(
            live, self.feature_b, task_id=9, learning_rate=0.01,
        )
        self.assertFalse(bool(updated))
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(norm), 0.0)
        self.assertTreeEqual(candidate, live)
        self.assertIs(source.snapshot, explicit)

    def test_candidate_encode_uses_explicit_snapshot_without_mutating_live_owner(self):
        initial = self.populate()
        empty = initial._replace(state=create_memory_state(4, 8))
        amplitudes = project_features(self.feature_b, empty.proj_real, empty.proj_imag)
        candidate = self.memory.candidate_encode(
            amplitudes, action_id=6, reward=2.0, task_id=4, phase=3.0,
            features=self.feature_b, snapshot=empty,
        )
        self.assertIsInstance(candidate, quantum_memory.JAXMemoryState)
        self.assertIs(self.memory.snapshot, initial)
        self.assertEqual(int(candidate.count), 1)
        self.assertEqual(int(candidate.total_encoded), 1)
        self.assertEqual(int(candidate.action_ids[0]), 6)
        np.testing.assert_array_equal(candidate.features[0], self.feature_b)
        legacy_candidate = self.memory.candidate_encode(
            amplitudes, action_id=4, reward=0.0, task_id=4, phase=4.0,
        )
        self.assertEqual(int(legacy_candidate.count), 4)
        self.assertIs(self.memory.snapshot, initial)

    def test_invalid_projection_rates_leave_snapshot_unchanged(self):
        initial = self.populate()
        for rate in (0.0, -0.01, float("nan"), float("inf"), -float("inf")):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    self.memory.candidate_projection(initial, self.feature_b, 9, rate)
                with self.assertRaises(ValueError):
                    self.memory.update_projection(self.feature_b, 9, rate)
                self.assertIs(self.memory.snapshot, initial)

    def test_update_projection_publishes_one_synchronized_candidate(self):
        initial = self.populate()
        candidate, loss, norm, updated = self.memory.candidate_projection(initial, self.feature_b, 9, 0.01)
        real_block_until_ready = jax.block_until_ready
        synchronized = []

        def synchronize(value):
            self.assertIs(self.memory.snapshot, initial)
            self.assertIn(id(candidate), [id(item) for item in value])
            synchronized.append(True)
            return real_block_until_ready(value)

        with mock.patch.object(self.memory, "candidate_projection", return_value=(candidate, loss, norm, updated)) as stage:
            with mock.patch.object(jax, "block_until_ready", side_effect=synchronize):
                result = self.memory.update_projection(self.feature_b, 9, 0.01)
        stage.assert_called_once_with(initial, self.feature_b, 9, 0.01)
        self.assertTrue(synchronized)
        self.assertIs(self.memory.snapshot, candidate)
        self.assertTreeEqual(result, (loss, norm, updated))

    def test_failed_projection_compute_or_synchronization_never_assigns(self):
        initial = self.populate()
        with mock.patch.object(self.memory, "candidate_projection", side_effect=RuntimeError("compute failed")):
            with self.assertRaisesRegex(RuntimeError, "compute failed"):
                self.memory.update_projection(self.feature_b, 9, 0.01)
        self.assertIs(self.memory.snapshot, initial)
        with mock.patch.object(jax, "block_until_ready", side_effect=RuntimeError("sync failed")):
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                self.memory.update_projection(self.feature_b, 9, 0.01)
        self.assertIs(self.memory.snapshot, initial)

    def test_nonfinite_candidate_or_diagnostics_never_assign(self):
        initial = self.populate()
        candidate, loss, norm, updated = self.memory.candidate_projection(initial, self.feature_b, 9, 0.01)
        nonfinite_candidates = (
            candidate._replace(proj_real=candidate.proj_real.at[0, 0].set(jnp.nan)),
            candidate._replace(proj_imag=candidate.proj_imag.at[0, 0].set(jnp.inf)),
            candidate._replace(state=candidate.state._replace(
                amplitudes=candidate.state.amplitudes.at[0, 0].set(jnp.complex64(complex(0, float("inf"))))
            )),
            candidate._replace(state=candidate.state._replace(rewards=candidate.state.rewards.at[0].set(jnp.nan))),
        )
        for invalid in nonfinite_candidates:
            with self.subTest(field="snapshot"):
                with mock.patch.object(self.memory, "candidate_projection", return_value=(invalid, loss, norm, updated)):
                    with self.assertRaises(FloatingPointError):
                        self.memory.update_projection(self.feature_b, 9, 0.01)
                self.assertIs(self.memory.snapshot, initial)
        for diagnostics in ((jnp.array(jnp.nan), norm), (loss, jnp.array(jnp.inf))):
            with self.subTest(field="diagnostic"):
                with mock.patch.object(self.memory, "candidate_projection", return_value=(candidate, *diagnostics, updated)):
                    with self.assertRaises(FloatingPointError):
                        self.memory.update_projection(self.feature_b, 9, 0.01)
                self.assertIs(self.memory.snapshot, initial)

    def test_binding_reads_current_owner_and_writes_under_shared_reentrant_lock(self):
        memory = self.memory
        standalone = memory.snapshot
        owner = [OwnerRuntime(standalone, marker=7)]
        lock = TrackingRLock()
        assignments = []

        def get_snapshot():
            self.assertGreater(lock.depth, 0)
            return owner[0].memory

        def set_snapshot(snapshot):
            self.assertGreater(lock.depth, 0)
            assignments.append(snapshot)
            owner[0] = dataclasses.replace(owner[0], memory=snapshot)

        memory.bind_snapshot(get_snapshot, set_snapshot, lock)
        self.assertIs(memory._lock, lock)
        replacement = standalone._replace(proj_real=standalone.proj_real + 0.25)
        with lock:
            owner[0] = OwnerRuntime(replacement, marker=11)
            self.assertIs(memory.snapshot, replacement)
            memory.state = replacement.state._replace(total_recalled=jnp.array(3, jnp.int32))
        self.assertEqual(owner[0].marker, 11)
        self.assertEqual(int(memory.state.total_recalled), 3)
        self.assertIs(memory._snapshot, standalone)
        memory.proj_real = standalone.proj_real
        memory.proj_imag = standalone.proj_imag
        memory.snapshot = standalone
        before_entries = lock.entries
        with lock:
            self.populate()
            features, amplitudes = memory.encode_grid_with_features(jnp.array([[0, 1], [1, 0]]))
            np.testing.assert_allclose(amplitudes, project_features(features, owner[0].memory.proj_real, owner[0].memory.proj_imag))
            memory.recall(amplitudes, phase=3.0, task_id=9)
            memory.update_projection(self.feature_b, task_id=9, learning_rate=0.01)
            stats = memory.get_stats()
        self.assertGreater(lock.entries, before_entries)
        self.assertEqual(stats["total_memories"], 3)
        self.assertEqual(stats["total_encoded"], 3)
        self.assertEqual(stats["total_recalled"], 1)
        self.assertEqual(owner[0].marker, 11)
        self.assertIs(memory.snapshot, owner[0].memory)
        self.assertIs(assignments[-1], owner[0].memory)
        self.assertIs(memory._snapshot, standalone)

    def test_mutating_operations_hold_shared_lock_through_computation(self):
        initial = self.populate()
        owner = [OwnerRuntime(initial)]
        lock = TrackingRLock()
        self.memory.bind_snapshot(
            lambda: owner[0].memory,
            lambda snapshot: owner.__setitem__(0, dataclasses.replace(owner[0], memory=snapshot)),
            lock,
        )

        def require_lock(function):
            def wrapped(*args, **kwargs):
                self.assertGreater(lock.depth, 0)
                return function(*args, **kwargs)
            return wrapped

        amplitudes = initial.state.amplitudes[0]
        with mock.patch.object(self.memory, "_encode", side_effect=require_lock(self.memory._encode)):
            self.memory.encode(amplitudes, 1, 0.0, 9, 3.0, features=self.feature_a)
        with mock.patch.object(self.memory, "_recall", side_effect=require_lock(self.memory._recall)):
            self.memory.recall(amplitudes, 4.0, 9)
        with mock.patch.object(self.memory, "candidate_projection", side_effect=require_lock(self.memory.candidate_projection)):
            with mock.patch.object(jax, "block_until_ready", side_effect=require_lock(jax.block_until_ready)):
                self.memory.update_projection(self.feature_b, 9, 0.01)
        with mock.patch.object(quantum_memory, "project_features", side_effect=require_lock(project_features)):
            self.memory.encode_grid_with_features(jnp.zeros((2, 2), jnp.int32))
        with mock.patch.object(jax, "device_get", side_effect=require_lock(jax.device_get)):
            self.memory.get_stats()

    def test_failed_bound_projection_never_calls_owner_setter(self):
        initial = self.populate()
        owner = [OwnerRuntime(initial, marker=19)]
        lock = TrackingRLock()
        publish = mock.Mock(side_effect=lambda snapshot: owner.__setitem__(
            0, dataclasses.replace(owner[0], memory=snapshot)
        ))
        self.memory.bind_snapshot(lambda: owner[0].memory, publish, lock)
        candidate, loss, norm, updated = self.memory.candidate_projection(initial, self.feature_b, 9, 0.01)
        invalid = candidate._replace(proj_real=candidate.proj_real.at[0, 0].set(jnp.nan))
        with mock.patch.object(self.memory, "candidate_projection", return_value=(invalid, loss, norm, updated)):
            with self.assertRaises(FloatingPointError):
                self.memory.update_projection(self.feature_b, 9, 0.01)
        publish.assert_not_called()
        self.assertIs(owner[0].memory, initial)
        self.assertEqual(owner[0].marker, 19)
        self.assertIs(self.memory.snapshot, initial)

    def test_grid_embedding_and_stats_capture_one_coherent_snapshot(self):
        populated = self.populate()
        empty = populated._replace(
            state=create_memory_state(4, 8),
            proj_real=populated.proj_real + 1.0,
            proj_imag=populated.proj_imag - 1.0,
        )
        lock = TrackingRLock()
        reads = []

        def get_snapshot():
            self.assertGreater(lock.depth, 0)
            reads.append(True)
            return populated if len(reads) == 1 else empty

        self.memory.bind_snapshot(get_snapshot, lambda snapshot: None, lock)
        reads.clear()
        features, amplitudes = self.memory.encode_grid_with_features(jnp.zeros((2, 2), jnp.int32))
        self.assertEqual(len(reads), 1)
        np.testing.assert_allclose(amplitudes, project_features(features, populated.proj_real, populated.proj_imag))
        reads.clear()
        stats = self.memory.get_stats()
        self.assertEqual(len(reads), 1)
        self.assertEqual(stats["total_memories"], 3)
        self.assertEqual(stats["total_encoded"], 3)
        self.assertEqual(stats["total_recalled"], 0)


if __name__ == "__main__":
    unittest.main()
