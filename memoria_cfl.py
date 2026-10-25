"""Memoria de replay de un cliente en el protocolo de varias tareas (cfl_tareas.py).

- Tamaño fijo (``capacity`` imágenes), repartido a partes iguales entre las clases vistas.
- Al terminar cada tarea se recorta cada clase antigua a su nueva cuota y se añaden las
  clases de la tarea recién aprendida. El cliente solo conserva la memoria: una imagen
  descartada no vuelve.
- Anidada: con la misma semilla, la memoria de capacidad menor está contenida en la de
  capacidad mayor después de cada tarea, porque
    * las cuotas se reparten por turnos sobre un orden de clases fijo del cliente
      (los primeros ``capacity mod n`` reciben una imagen más), y
    * cada clase guarda un prefijo de un orden de imágenes fijo.
  Con ese orden, la cuota de una clase antigua nunca crece de una tarea a la siguiente;
  aun así se acota por la anterior, por si una clase con pocas imágenes lo alterara.
"""

import numpy as np
import torch


def class_quotas(capacity, available, order):
    """Reparte ``capacity`` por turnos entre las clases de ``order``, sin pasar de ``available[c]``."""
    quotas = {c: 0 for c in order}
    remaining = min(capacity, sum(available[c] for c in order))
    while remaining > 0:
        for c in order:
            if remaining and quotas[c] < available[c]:
                quotas[c] += 1
                remaining -= 1
    return quotas


class ClientMemory:
    def __init__(self, capacity, seed, client, num_classes):
        self.capacity, self.seed, self.client = capacity, seed, client
        self.class_order = np.random.default_rng([seed, client]).permutation(num_classes).tolist()
        self.stored = {}  # clase -> imágenes guardadas (uint8), en su orden fijo
        self.available = {}  # clase -> imágenes que tenía el cliente al aprender la clase

    def add_task(self, x, y):
        """Incorpora la tarea recién aprendida (x, y) y recorta las clases antiguas."""
        new = {}
        for c in sorted(set(y.tolist())):
            idx = torch.nonzero(y == c).flatten()
            order = np.random.default_rng([self.seed, self.client, c]).permutation(len(idx))
            new[c] = x[idx[torch.as_tensor(order)]]
            self.available[c] = len(idx)
        seen = [c for c in self.class_order if c in self.available]
        quotas = class_quotas(self.capacity, self.available, seen)
        for c in seen:  # una clase antigua solo puede perder imágenes (y una descartada no vuelve)
            if c not in new:
                quotas[c] = min(quotas[c], len(self.stored.get(c, ())))
        self.stored = {c: (new[c] if c in new else self.stored[c])[:quotas[c]] for c in seen if quotas[c] > 0}

    def size(self):
        return sum(len(v) for v in self.stored.values())

    def quotas(self):
        return {c: len(v) for c, v in self.stored.items()}

    def tensors(self):
        """(x, y) de toda la memoria, o None si está vacía."""
        if not self.stored:
            return None
        classes = sorted(self.stored)
        x = torch.cat([self.stored[c] for c in classes])
        y = torch.cat([torch.full((len(self.stored[c]),), c, dtype=torch.long) for c in classes])
        return x, y


def memory_capacity(percent, client_samples, num_tasks):
    """Capacidad fija de un cliente: ``percent`` % de sus imágenes de una tarea media."""
    return percent * client_samples // (100 * num_tasks)
