"""Muestreo reproducible para las comparaciones controladas de CIFAR-100."""

import numpy as np


def balanced_order(labels, seed):
    """Ordena índices por turnos de clase; sus prefijos forman buffers anidados."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels))
    buckets = [rng.permutation(np.flatnonzero(labels == label)) for label in classes]
    ordered = []
    for offset in range(max((len(bucket) for bucket in buckets), default=0)):
        ordered.extend(int(bucket[offset]) for bucket in buckets if offset < len(bucket))
    return ordered


def balanced_subset(dataset, limit, seed):
    column = "fine_label" if "fine_label" in dataset.column_names else "label"
    if limit < len(set(dataset[column])):
        raise ValueError("El subconjunto no puede cubrir todas las clases con ese límite")
    return dataset.select(balanced_order(dataset[column], seed)[:limit])
