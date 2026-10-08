"""Pruebas del núcleo común (FedAvg, lotes de replay, preprocesado, métricas)."""

import unittest
from collections import Counter

import torch
import torch.nn as nn

import fl_core
from cifar100_validation import validate_training_contract


class FedAvgTests(unittest.TestCase):
    def test_mean_of_float_parameters(self):
        a = {"w": torch.tensor([1.0, 2.0]), "n": torch.tensor(3)}
        b = {"w": torch.tensor([3.0, 6.0]), "n": torch.tensor(4)}
        avg = fl_core.fedavg([a, b])
        torch.testing.assert_close(avg["w"], torch.tensor([2.0, 4.0]))
        self.assertEqual(avg["w"].dtype, torch.float32)
        # Los enteros (num_batches_tracked) se redondean, no se truncan: 3.5 -> 4.
        self.assertEqual(avg["n"].item(), 4)
        self.assertEqual(avg["n"].dtype, torch.int64)

    def test_weighted_mean(self):
        avg = fl_core.fedavg([{"w": torch.tensor(0.0)}, {"w": torch.tensor(10.0)}], weights=[3, 1])
        self.assertAlmostEqual(avg["w"].item(), 2.5)

    def test_identical_clients_leave_model_unchanged(self):
        model = fl_core.get_model(10)
        state = model.state_dict()
        avg = fl_core.fedavg([state, state, state])
        for key in state:
            torch.testing.assert_close(avg[key], state[key])

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            fl_core.fedavg([])
        with self.assertRaises(ValueError):
            fl_core.fedavg([{"w": torch.tensor(1.0)}], weights=[0])


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.new_x = torch.zeros(100, 3, 4, 4, dtype=torch.uint8)
        self.new_y = torch.full((100,), 70)
        self.buf_x = torch.ones(10, 3, 4, 4, dtype=torch.uint8)
        self.buf_y = torch.full((10,), 5)

    def test_mixed_batches_have_fixed_replay_share_and_step_count(self):
        gen = torch.Generator().manual_seed(0)
        batches = list(fl_core.mixed_batches(self.new_x, self.new_y, self.buf_x, self.buf_y, 32, 7, 0.25, gen))
        self.assertEqual(len(batches), 7)
        for _, y in batches:
            self.assertEqual(len(y), 32)
            self.assertEqual((y == 5).sum().item(), 8)

    def test_mixed_batches_without_buffer_only_use_new_data(self):
        batches = list(fl_core.mixed_batches(self.new_x, self.new_y, None, None, 32, 3, 0.5))
        self.assertTrue(all((y == 70).all() for _, y in batches))

    def test_concat_batches_respect_max_steps_and_cover_epoch(self):
        gen = torch.Generator().manual_seed(0)
        batches = list(fl_core.concat_batches(self.new_x, self.new_y, self.buf_x, self.buf_y, 32, 2, generator=gen))
        self.assertEqual(sum(len(y) for _, y in batches), 2 * 110)
        capped = list(fl_core.concat_batches(self.new_x, self.new_y, None, None, 32, 5, max_steps=6))
        self.assertEqual(len(capped), 6)

    def test_same_generator_seed_gives_same_batches(self):
        run = lambda: [y.tolist() for _, y in fl_core.concat_batches(
            torch.zeros(20, 1), torch.arange(20), None, None, 5, 1, generator=torch.Generator().manual_seed(3))]
        self.assertEqual(run(), run())


class PreprocessTests(unittest.TestCase):
    def test_normalization_and_shape(self):
        pre = fl_core.Preprocess(normalize=True)
        x = torch.full((2, 3, 32, 32), 255, dtype=torch.uint8)
        out = pre(x)
        expected = (1 - torch.tensor(fl_core.CIFAR100_MEAN)) / torch.tensor(fl_core.CIFAR100_STD)
        torch.testing.assert_close(out[0, :, 0, 0], expected)
        self.assertEqual(pre(x, augment=True, generator=torch.Generator().manual_seed(1)).shape, x.shape)

    def test_without_normalization_matches_old_protocol(self):
        x = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        torch.testing.assert_close(fl_core.Preprocess(normalize=False)(x), x.float() / 255)

    def test_augmentation_does_not_touch_global_rng(self):
        torch.manual_seed(0)
        expected = torch.rand(3)
        torch.manual_seed(0)
        fl_core.Preprocess()(torch.zeros(4, 3, 8, 8, dtype=torch.uint8), augment=True,
                             generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(torch.rand(3), expected)


class EvaluateTests(unittest.TestCase):
    def test_group_metrics_detect_bias_towards_new_classes(self):
        class AlwaysB(nn.Module):
            def forward(self, x):
                logits = torch.zeros(len(x), 4)
                logits[:, 3] = 10       # siempre predice la clase 3 (grupo B)
                logits[:, 1] = 5        # la clase A más probable es la 1
                return logits

        x = torch.zeros(4, 3, 2, 2)
        y = torch.tensor([0, 1, 1, 1])  # todo grupo A
        result = fl_core.evaluate(AlwaysB(), x, y, "cpu", group_a_classes=[0, 1])
        self.assertEqual(result["accuracy_percent"], 0)
        self.assertEqual(result["pred_share_b_percent"], 100)
        self.assertEqual(result["group_aware_accuracy_percent"], 75)

    def test_retention(self):
        self.assertEqual(fl_core.retention_percent(50, 25), 50)
        self.assertIsNone(fl_core.retention_percent(0, 0))


class TrainTests(unittest.TestCase):
    def test_train_local_records_steps_and_updates_weights(self):
        torch.manual_seed(0)
        model = fl_core.get_model(10)
        x = torch.randint(0, 256, (16, 3, 8, 8), dtype=torch.uint8)
        y = torch.randint(0, 10, (16,))
        record = {}
        fl_core.train_local(model, fl_core.concat_batches(x, y, None, None, 8, 2), 1e-3, "cpu",
                            fl_core.Preprocess(), record=record)
        self.assertEqual(record["optimizer_steps"], 4)
        self.assertEqual(record["samples_processed"], 32)
        self.assertGreater(record["fc_weight_delta_l2"], 0)

    def test_distillation_is_zero_for_identical_models(self):
        logits = torch.randn(5, 10)
        self.assertAlmostEqual(fl_core.distillation_loss(logits, logits, range(5)).item(), 0, places=6)


class ContractTests(unittest.TestCase):
    def test_old_manifest_implies_no_normalization(self):
        old = {"num_clients": 10, "seed": 42, "dataset_revision": "r"}
        with self.assertRaisesRegex(ValueError, "normalize"):
            validate_training_contract({**old, "normalize": True, "augment": True}, old)
        validate_training_contract({**old, "normalize": False, "augment": False}, old)


if __name__ == "__main__":
    unittest.main()
