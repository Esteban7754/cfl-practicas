"""Comprobaciones de cobertura, anidación y reproducibilidad del muestreo."""

import unittest
from collections import Counter

import numpy as np

from cifar100_sampling import balanced_order
from cifar100_validation import learning_problem, validate_training_contract


class SamplingTests(unittest.TestCase):
    def test_checkpoint_full_pool_cannot_be_used_as_reduced_replay_pool(self):
        full = {"num_clients": 10, "seed": 42, "dataset_revision": "revision"}
        reduced = {**full, "num_clients": 2, "train_samples_per_group": 500, "sampling": "balanced"}
        with self.assertRaisesRegex(ValueError, "train_samples_per_group"):
            validate_training_contract(reduced, full)
        validate_training_contract(full, full)
        validate_training_contract(reduced, reduced)

    def test_pilot_rejects_training_failure_even_when_test_floor_passes(self):
        config = {"min_new_task_accuracy_percent": 8, "min_training_accuracy_percent": 50}
        self.assertIsNotNone(learning_problem(9.1, 12, None, config))
        self.assertIsNotNone(learning_problem(20, 80, 0, config))
        self.assertIsNone(learning_problem(20, 80, 75, config))
        self.assertIsNone(learning_problem(20, 80, None, config))

    def test_balanced_prefixes_cover_every_class_and_are_nested(self):
        labels = np.repeat(np.arange(50), 10)
        indices = balanced_order(labels, 42)
        self.assertEqual(len(indices), 500)
        self.assertEqual(set(indices), set(range(500)))
        for size in (50, 100, 500):
            self.assertEqual(set(Counter(labels[indices[:size]]).values()), {size // 50})

    def test_unbalanced_input_keeps_every_sample_without_duplicates(self):
        labels = [0] * 3 + [1] * 7 + [2]
        indices = balanced_order(labels, 42)
        self.assertEqual(sorted(indices), list(range(len(labels))))
        self.assertEqual(set(np.array(labels)[indices[:3]]), {0, 1, 2})

    def test_own_random_generator_does_not_change_global_state(self):
        np.random.seed(123)
        expected = np.random.random(3)
        np.random.seed(123)
        first = balanced_order(np.repeat(np.arange(50), 10), 42)
        np.testing.assert_array_equal(np.random.random(3), expected)
        self.assertEqual(first, balanced_order(np.repeat(np.arange(50), 10), 42))
        self.assertNotEqual(first, balanced_order(np.repeat(np.arange(50), 10), 43))


if __name__ == "__main__":
    unittest.main()
