import importlib.machinery
import importlib.util
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if 'adjudicator' not in sys.modules:
    loader = importlib.machinery.SourceFileLoader('adjudicator', str(ROOT / 'adjudicator'))
    spec = importlib.util.spec_from_loader('adjudicator', loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules['adjudicator'] = module
    loader.exec_module(module)
adjudicator = sys.modules['adjudicator']


def config():
    return adjudicator.SwarmConfig(
        max_agents=2, min_agents=2, chunk_size=2, grid_side=4, hidden_dim=8,
        graph_capacity=8, retrieval_top_k=2, quantum_memory_dim=8,
        quantum_memory_capacity=8, quantum_memory_top_k=2, num_roles=2,
        qpso_particles=2, qpso_every=1, rotation_every=1,
        quantum_projection_update_every=1, seed=11,
    )


def learning_snapshot(swarm):
    values = (swarm.params, swarm.state, swarm.quantum_memory.state,
              swarm.quantum_memory.proj_real, swarm.quantum_memory.proj_imag,
              swarm.qpso, swarm.role_fitness, swarm.prev_role_fitness,
              swarm._qpso_key, swarm._qpso_steps, swarm._surprise_mean,
              swarm._surprise_var, swarm._surprise_count)
    return [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(values)]


class LearningStateAssertions(unittest.TestCase):
    def assertLearningEqual(self, a, b):
        self.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)


class FeedbackTransactionTests(LearningStateAssertions):
    def warm_up(self, swarm):
        grid = np.eye(4, dtype=np.int32)
        swarm.propose(grid, [1, 2, 6], 3)
        swarm.observe_outcome(0.2, grid, task_id=3)

    def test_projection_failure_does_not_commit_and_retry_matches_clean_update(self):
        broken = adjudicator.JAXProposalSwarm(config())
        clean = adjudicator.JAXProposalSwarm(config())
        for swarm in (broken, clean):
            self.warm_up(swarm)
            swarm.propose(np.ones((4, 4), np.int32), [1, 2, 6], 3)
        before = learning_snapshot(broken)
        pending = broken._pending
        with patch.object(broken.quantum_memory, '_projection_train',
                          side_effect=RuntimeError('projection failed')):
            with self.assertRaisesRegex(RuntimeError, 'projection failed'):
                broken.observe_outcome(1.0, np.zeros((4, 4), np.int32), 3)
        self.assertLearningEqual(before, learning_snapshot(broken))
        self.assertIs(broken._pending, pending)
        result = broken.observe_outcome(1.0, np.zeros((4, 4), np.int32), 3)
        expected = clean.observe_outcome(1.0, np.zeros((4, 4), np.int32), 3)
        self.assertEqual(result, expected)
        self.assertLearningEqual(learning_snapshot(broken), learning_snapshot(clean))

    def test_late_stage_failures_leave_every_field_unchanged(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        self.warm_up(swarm)
        swarm.propose(np.ones((4, 4), np.int32), [1, 2], 3)
        for target, owner in [('role_rotation_update', adjudicator),
                              ('qpso_step', adjudicator),
                              ('_synchronize_feedback', swarm),
                              ('_feedback_result', swarm)]:
            with self.subTest(stage=target):
                before = learning_snapshot(swarm)
                pending = swarm._pending
                with patch.object(owner, target, create=True,
                                  side_effect=RuntimeError(target)):
                    with self.assertRaisesRegex(RuntimeError, target):
                        swarm.observe_outcome(0.5, np.zeros((4, 4), np.int32), 3)
                self.assertLearningEqual(before, learning_snapshot(swarm))
                self.assertIs(swarm._pending, pending)

    def test_duplicate_delivery_returns_cached_result_without_learning(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)
        decision = swarm.propose(grid, [1, 2], 3)
        feedback = swarm.observe_outcome(0.5, grid, 3, proposal_id=decision.proposal_id)
        before = learning_snapshot(swarm)
        feedback['step'] = -100  # caller cannot mutate the cached result
        replay = swarm.observe_outcome(0.5, grid.copy(), 3, proposal_id=decision.proposal_id)
        self.assertEqual(replay['step'], 1)
        self.assertLearningEqual(before, learning_snapshot(swarm))
        with self.assertRaises(ValueError):
            swarm.observe_outcome(0.6, grid, 3, proposal_id=decision.proposal_id)
        with self.assertRaises(ValueError):
            swarm.observe_outcome(0.5, grid + 1, 3, proposal_id=decision.proposal_id)
        with self.assertRaises(ValueError):
            swarm.observe_outcome(0.5, grid, 4, proposal_id=decision.proposal_id)
        self.assertLearningEqual(before, learning_snapshot(swarm))

    def test_old_duplicate_cannot_consume_a_new_proposal(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)
        first = swarm.propose(grid, [1, 2], 3)
        old = swarm.observe_outcome(0.5, grid, 3, proposal_id=first.proposal_id)
        second = swarm.propose(grid, [1, 2], 3)
        self.assertNotEqual(first.proposal_id, second.proposal_id)
        before = learning_snapshot(swarm)
        self.assertEqual(swarm.observe_outcome(0.5, grid, 3, proposal_id=first.proposal_id), old)
        self.assertLearningEqual(before, learning_snapshot(swarm))
        self.assertIsNotNone(swarm._pending)
        swarm.observe_outcome(0.5, grid, 3, proposal_id=second.proposal_id)
        self.assertEqual(int(swarm.state.step_count), 2)

    def test_unknown_or_cancelled_id_cannot_learn(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)
        first = swarm.propose(grid, [1, 2], 3)
        before = learning_snapshot(swarm)
        with self.assertRaises(ValueError):
            swarm.observe_outcome(0.5, grid, 3, proposal_id='unknown')
        self.assertLearningEqual(before, learning_snapshot(swarm))
        swarm.cancel_pending()
        swarm.propose(grid, [1, 2], 3)
        with self.assertRaises(ValueError):
            swarm.observe_outcome(0.5, grid, 3, proposal_id=first.proposal_id)

    def test_nonfinite_candidate_is_rejected_before_commit(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        swarm.propose(np.ones((4, 4), np.int32), [1, 2], 3)
        before = learning_snapshot(swarm)
        original = swarm._feedback_jit

        def invalid(*args):
            params, state, error, surprise, diagnostics = original(*args)
            return params._replace(agent_action_bias=params.agent_action_bias * np.nan), state, error, surprise, diagnostics

        with patch.object(swarm, '_feedback_jit', side_effect=invalid):
            with self.assertRaises(FloatingPointError):
                swarm.observe_outcome(0.5, task_id=3)
        self.assertLearningEqual(before, learning_snapshot(swarm))

    def test_concurrent_duplicates_commit_once(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)
        decision = swarm.propose(grid, [1, 2], 3)
        barrier = threading.Barrier(2)

        def deliver():
            barrier.wait()
            return swarm.observe_outcome(0.5, grid, 3, proposal_id=decision.proposal_id)

        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = [executor.submit(deliver) for _ in range(2)]
            results = [job.result(timeout=60) for job in jobs]
        self.assertEqual(results[0], results[1])
        self.assertEqual(int(swarm.state.step_count), 1)
        self.assertEqual(int(swarm.quantum_memory.state.total_encoded), 1)
        self.assertEqual(swarm._surprise_count, 1)

    def test_delayed_qpso_preserves_unscored_fitness_until_first_generation(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        swarm.qpso_every = 3
        grid = np.ones((4, 4), np.int32)
        for step in range(1, 4):
            swarm.propose(grid, [1, 2], 3)
            result = swarm.observe_outcome(0.5, grid, 3)
            self.assertEqual(result['step'], step)
            if step < 3:
                self.assertEqual(result['qpso_generation'], 0)
                self.assertTrue(np.isneginf(result['qpso_best_fitness']))
            else:
                self.assertEqual(result['qpso_generation'], 1)
                self.assertTrue(np.isfinite(result['qpso_best_fitness']))

    def test_memory_insertion_failure_does_not_advance_surprise_history(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        swarm.propose(np.ones((4, 4), np.int32), [1, 2], 3)
        before = learning_snapshot(swarm)
        with patch.object(swarm.quantum_memory, '_encode', side_effect=RuntimeError('insert')):
            with self.assertRaisesRegex(RuntimeError, 'insert'):
                swarm.observe_outcome(0.5, task_id=3)
        self.assertLearningEqual(before, learning_snapshot(swarm))
        result = swarm.observe_outcome(0.5, task_id=3)
        self.assertEqual(result['step'], 1)
        self.assertEqual(swarm._surprise_count, 1)

    def test_completion_record_survives_episode_reset(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)
        decision = swarm.propose(grid, [1, 2], 3)
        result = swarm.observe_outcome(0.5, grid, 3, proposal_id=decision.proposal_id)
        swarm.reset_episode(preserve_graph=False)
        before = learning_snapshot(swarm)
        self.assertEqual(swarm.observe_outcome(0.5, grid, proposal_id=decision.proposal_id), result)
        self.assertLearningEqual(before, learning_snapshot(swarm))


class ExecutionBoundaryTests(LearningStateAssertions):
    def test_feedback_retry_does_not_execute_again(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)
        executions = []

        def execute(action, coordinates):
            executions.append(action)
            return {'reward': 0.5}

        with patch.object(swarm.quantum_memory, '_projection_train', side_effect=RuntimeError('learn')):
            with self.assertRaisesRegex(RuntimeError, 'learn'):
                swarm.step(grid, [1, 2], 3, execute, lambda raw: (raw['reward'], grid))
        self.assertIsNotNone(swarm._pending)
        self.assertEqual(int(swarm.state.step_count), 0)
        self.assertEqual(int(swarm.quantum_memory.state.count), 0)
        with self.assertRaises(RuntimeError):
            swarm.cancel_pending()
        with self.assertRaises(RuntimeError):
            swarm.step(grid, [1, 2], 3, execute, lambda raw: (raw['reward'], grid))
        decision, raw, result = swarm.retry_feedback()
        self.assertEqual(len(executions), 1)
        self.assertEqual(raw, {'reward': 0.5})
        self.assertEqual(result['step'], 1)
        replay = swarm.observe_outcome(0.5, grid, 3, proposal_id=decision.proposal_id)
        self.assertEqual(replay, result)

    def test_parse_failure_retains_raw_receipt(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)
        receipt = object()
        executions = []

        def execute(action, coordinates):
            executions.append(action)
            return receipt

        def bad_parser(raw):
            raise ValueError('parse')

        with self.assertRaisesRegex(ValueError, 'parse'):
            swarm.step(grid, [1, 2], 3, execute, bad_parser)
        _, raw, result = swarm.retry_feedback(parse_outcome=lambda raw: (0.5, grid))
        self.assertIs(raw, receipt)
        self.assertEqual(result['step'], 1)
        self.assertEqual(len(executions), 1)

    def test_execution_error_is_uncertain_and_requires_reconciliation(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)

        def uncertain(action, coordinates):
            raise OSError('unknown outcome')

        with self.assertRaisesRegex(OSError, 'unknown outcome'):
            swarm.step(grid, [1, 2], 3, uncertain, lambda raw: (0.5, grid))
        self.assertIsNotNone(swarm._pending)
        with self.assertRaises(RuntimeError):
            swarm.retry_feedback()
        with self.assertRaises(RuntimeError):
            swarm.cancel_pending()
        result = swarm.observe_outcome(0.5, grid, 3, proposal_id=swarm.last_decision.proposal_id)
        self.assertEqual(result['step'], 1)

    def test_retained_outcome_rejects_conflicting_parser_and_direct_feedback(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)
        with patch.object(swarm.quantum_memory, '_projection_train', side_effect=RuntimeError('learn')):
            with self.assertRaisesRegex(RuntimeError, 'learn'):
                swarm.step(grid, [1, 2], 3, lambda action, xy: object(), lambda raw: (0.5, grid))
        before = learning_snapshot(swarm)
        with self.assertRaises(ValueError):
            swarm.retry_feedback(parse_outcome=lambda raw: (0.6, grid))
        with self.assertRaises(ValueError):
            swarm.observe_outcome(0.6, grid, 3)
        self.assertLearningEqual(before, learning_snapshot(swarm))
        _, _, result = swarm.retry_feedback()
        self.assertEqual(result['step'], 1)

    def test_explicit_abandonment_drops_pending_learning_not_external_action(self):
        swarm = adjudicator.JAXProposalSwarm(config())
        grid = np.ones((4, 4), np.int32)
        with self.assertRaises(ValueError):
            swarm.step(grid, [1, 2], 3, lambda action, xy: None,
                       lambda raw: (_ for _ in ()).throw(ValueError('parse')))
        proposal_id = swarm.last_decision.proposal_id
        swarm.cancel_pending(abandon_executed=True)
        self.assertIsNone(swarm._pending)
        self.assertEqual(int(swarm.state.step_count), 0)
        self.assertEqual(int(swarm.quantum_memory.state.count), 0)
        with self.assertRaises(ValueError):
            swarm.observe_outcome(0.5, grid, 3, proposal_id=proposal_id)


if __name__ == '__main__':
    unittest.main()
