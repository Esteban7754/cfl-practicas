"""Tests de la memoria de replay por clases (memoria_cfl.py). Solo torch y numpy."""

import unittest

import torch

from memoria_cfl import ClientMemory, class_quotas, memory_capacity


def task_data(task, per_class=50, classes_per_task=20):
    """Tarea con imágenes identificables: cada «imagen» es su número de muestra global."""
    classes = range(task * classes_per_task, (task + 1) * classes_per_task)
    y = torch.tensor([c for c in classes for _ in range(per_class)])
    x = (torch.arange(len(y)) + 100_000 * task).view(-1, 1)
    return x, y


def run_tasks(capacity, seed=42, tasks=5, **kwargs):
    memory, snapshots = ClientMemory(capacity, seed, client=3, num_classes=100), []
    for t in range(tasks):
        memory.add_task(*task_data(t, **kwargs))
        snapshots.append({c: v.clone() for c, v in memory.stored.items()})
    return memory, snapshots


class QuotaTests(unittest.TestCase):
    def test_round_robin_respects_order_and_caps(self):
        self.assertEqual(class_quotas(7, {1: 9, 2: 9, 3: 9}, [2, 1, 3]), {2: 3, 1: 2, 3: 2})
        self.assertEqual(class_quotas(7, {1: 1, 2: 9, 3: 9}, [1, 2, 3]), {1: 1, 2: 3, 3: 3})
        self.assertEqual(class_quotas(50, {1: 2, 2: 3}, [1, 2]), {1: 2, 2: 3})

    def test_capacity_is_a_percentage_of_an_average_task(self):
        self.assertEqual(memory_capacity(20, 5000, 5), 200)
        self.assertEqual(memory_capacity(10, 5000, 5), 100)
        self.assertEqual(memory_capacity(0, 5000, 5), 0)


class ClientMemoryTests(unittest.TestCase):
    def test_fixed_size_split_evenly_among_seen_classes(self):
        for capacity in (100, 200):
            _, snapshots = run_tasks(capacity)
            for t, snap in enumerate(snapshots):
                quotas = [len(v) for v in snap.values()]
                self.assertEqual(sum(quotas), capacity, (capacity, t))
                self.assertLessEqual(max(quotas) - min(quotas), 1)
                self.assertEqual(sorted(snap), list(range(20 * (t + 1))))  # todas las clases vistas

    def test_old_classes_only_lose_images(self):
        _, snapshots = run_tasks(100)
        for before, after in zip(snapshots, snapshots[1:]):
            for c, images in after.items():
                if c in before:
                    self.assertTrue(torch.equal(images, before[c][:len(images)]))

    def test_smaller_memory_is_contained_in_larger_one(self):
        _, small = run_tasks(100)
        _, large = run_tasks(200)
        for s, l in zip(small, large):
            for c, images in s.items():
                self.assertTrue(torch.equal(images, l[c][:len(images)]))

    def test_tiny_memory_keeps_at_most_one_image_per_class(self):
        memory, _ = run_tasks(30)
        self.assertEqual(memory.size(), 30)
        self.assertEqual(set(memory.quotas().values()), {1})

    def test_scarce_classes_leave_room_for_the_rest(self):
        memory = ClientMemory(40, 42, client=0, num_classes=100)
        y = torch.tensor([0] + [1] * 50 + [2] * 50)
        memory.add_task(torch.arange(len(y)).view(-1, 1), y)
        self.assertEqual(memory.quotas()[0], 1)
        self.assertEqual(memory.size(), 40)

    def test_tensors_match_stored_images(self):
        memory, _ = run_tasks(100, tasks=2)
        x, y = memory.tensors()
        self.assertEqual(len(y), memory.size())
        self.assertEqual(set(y.tolist()), set(range(40)))
        self.assertIsNone(ClientMemory(0, 42, 0, 100).tensors())

    def test_same_seed_same_memory(self):
        _, a = run_tasks(100)
        _, b = run_tasks(100)
        self.assertTrue(all(torch.equal(a[-1][c], b[-1][c]) for c in a[-1]))


if __name__ == "__main__":
    unittest.main()
