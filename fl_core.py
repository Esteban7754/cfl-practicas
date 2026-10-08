"""Núcleo común de los experimentos federados continuos.

Antes cada script tenía su propia copia de ``get_model``, ``train_local``,
``evaluate`` y FedAvg. Este módulo centraliza esas piezas y añade:

* preprocesado sobre tensores en memoria (mucho más rápido que transformar
  cada imagen PIL en cada época) con normalización y aumento de datos opcionales;
* FedAvg que trata bien los buffers enteros de BatchNorm;
* lotes de replay con proporción fija de muestras antiguas (``mixed_batches``);
* destilación opcional sobre los logits de las clases antiguas (estilo LwF);
* métricas que no se quedan en el «suelo» de la precisión: precisión con el
  grupo conocido, reparto de predicciones entre grupos y entropía cruzada.

No importa nada al cargarse más allá de PyTorch/NumPy, así que se puede probar
con ``python -m unittest test_fl_core``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# Estadísticas estándar del conjunto de entrenamiento de CIFAR-100.
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------
# Modelo y agregación
# ------------------------------------------------------------------
def get_model(num_classes: int = 100) -> nn.Module:
    """ResNet-18 adaptada a imágenes pequeñas (conv 3x3, sin max-pool inicial)."""
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def fedavg(state_dicts: Sequence[dict], weights: Optional[Sequence[float]] = None) -> dict:
    """Media (opcionalmente ponderada) de los parámetros de los clientes.

    Los tensores enteros (``num_batches_tracked`` de BatchNorm) se redondean en
    vez de truncarse al volver a su tipo original.
    """
    if not state_dicts:
        raise ValueError("FedAvg necesita al menos un cliente")
    if weights is None:
        weights = [1.0] * len(state_dicts)
    if len(weights) != len(state_dicts) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Pesos de FedAvg no válidos")
    total = float(sum(weights))
    averaged = {}
    for key in state_dicts[0]:
        reference = state_dicts[0][key]
        stacked = torch.stack([sd[key].detach().to(torch.float64) for sd in state_dicts], dim=0)
        coeffs = torch.tensor([w / total for w in weights], dtype=torch.float64,
                              device=stacked.device).view(-1, *([1] * reference.dim()))
        mean = (stacked * coeffs).sum(dim=0)
        if reference.is_floating_point():
            averaged[key] = mean.to(reference.dtype)
        else:
            averaged[key] = mean.round().to(reference.dtype)
    return averaged


def weights_digest(model: nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------
# Datos en memoria y preprocesado
# ------------------------------------------------------------------
def hf_to_tensors(dataset, image_column: str = "img", label_column: Optional[str] = None):
    """Convierte un dataset de Hugging Face en (imágenes uint8 NCHW, etiquetas int64)."""
    if label_column is None:
        label_column = "fine_label" if "fine_label" in dataset.column_names else "label"
    if len(dataset) == 0:
        return torch.zeros((0, 3, 32, 32), dtype=torch.uint8), torch.zeros(0, dtype=torch.long)
    images = np.stack([np.asarray(img, dtype=np.uint8) for img in dataset[image_column]])
    if images.ndim == 3:  # escala de grises
        images = np.repeat(images[..., None], 3, axis=-1)
    x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(dataset[label_column], dtype=torch.long)
    return x, y


class Preprocess:
    """uint8 -> float [0, 1], aumento opcional y normalización opcional.

    El aumento (recorte aleatorio con relleno de 4 píxeles y volteo horizontal)
    usa su propio ``torch.Generator`` para no alterar el estado global.
    """

    def __init__(self, normalize: bool = True, mean=CIFAR100_MEAN, std=CIFAR100_STD, pad: int = 4):
        self.normalize = normalize
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.pad = pad

    def __call__(self, x: torch.Tensor, augment: bool = False, generator: Optional[torch.Generator] = None):
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        if augment:
            x = self.augment(x, generator)
        if self.normalize:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x

    def augment(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        n, _, h, w = x.shape
        if n == 0:
            return x
        p = self.pad
        padded = F.pad(x, (p, p, p, p))
        dy = torch.randint(0, 2 * p + 1, (n,), generator=generator)
        dx = torch.randint(0, 2 * p + 1, (n,), generator=generator)
        flip = torch.rand(n, generator=generator) < 0.5
        out = torch.empty_like(x)
        for i in range(n):
            crop = padded[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w]
            out[i] = crop.flip(-1) if flip[i] else crop
        return out

    def describe(self) -> dict:
        return {"normalize": self.normalize, "mean": self.mean.flatten().tolist(),
                "std": self.std.flatten().tolist(), "augment_pad": self.pad}


def shuffled_batches(x, y, batch_size, generator=None, shuffle=True) -> Iterator:
    order = torch.randperm(len(y), generator=generator) if shuffle else torch.arange(len(y))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]


def concat_batches(new_x, new_y, buffer_x, buffer_y, batch_size, epochs, max_steps=None,
                   generator=None) -> Iterator:
    """Diseño original: se barajan juntos datos nuevos y buffer, época a época."""
    if buffer_x is not None and len(buffer_y):
        x, y = torch.cat([new_x, buffer_x]), torch.cat([new_y, buffer_y])
    else:
        x, y = new_x, new_y
    steps = 0
    for _ in range(epochs):
        for batch in shuffled_batches(x, y, batch_size, generator):
            yield batch
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return


def mixed_batches(new_x, new_y, buffer_x, buffer_y, batch_size, steps, replay_fraction,
                  generator=None) -> Iterator:
    """Lotes con una proporción fija de muestras del buffer.

    En cada lote, ``round(batch_size * replay_fraction)`` muestras salen del
    buffer (con reposición, porque suele ser más pequeño) y el resto de los
    datos nuevos, recorridos por épocas barajadas. Así se separa el *tamaño*
    del buffer de la *proporción* de replay en cada actualización.
    """
    if not 0 <= replay_fraction < 1:
        raise ValueError("replay_fraction debe estar en [0, 1)")
    n_old = int(round(batch_size * replay_fraction)) if buffer_x is not None and len(buffer_y) else 0
    n_new = batch_size - n_old
    order = torch.randperm(len(new_y), generator=generator)
    cursor = 0
    for _ in range(steps):
        if cursor + n_new > len(order):
            order = torch.randperm(len(new_y), generator=generator)
            cursor = 0
        idx_new = order[cursor:cursor + n_new]
        cursor += n_new
        xb, yb = new_x[idx_new], new_y[idx_new]
        if n_old:
            idx_old = torch.randint(0, len(buffer_y), (n_old,), generator=generator)
            xb, yb = torch.cat([xb, buffer_x[idx_old]]), torch.cat([yb, buffer_y[idx_old]])
        yield xb, yb


def balanced_buffer_order(labels, seed: int):
    """Orden por turnos de clase cuyos prefijos forman buffers anidados."""
    from cifar100_sampling import balanced_order

    return balanced_order(np.asarray(labels), seed)


# ------------------------------------------------------------------
# Entrenamiento local y evaluación
# ------------------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, old_classes: Sequence[int], temperature: float = 2.0):
    """KL entre las distribuciones del modelo anterior y el actual sobre las clases antiguas."""
    idx = torch.as_tensor(list(old_classes), device=student_logits.device)
    s = F.log_softmax(student_logits.index_select(1, idx) / temperature, dim=1)
    t = F.softmax(teacher_logits.index_select(1, idx) / temperature, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * temperature ** 2


def train_local(model, batches, lr, device, preprocess: Optional[Callable] = None, augment=False,
                generator=None, teacher=None, distill_weight=0.0, old_classes=None,
                record: Optional[dict] = None):
    """Entrena con Adam sobre un iterable de lotes y devuelve el ``state_dict``.

    ``record`` (opcional) recibe pasos, pérdidas, etiquetas vistas y el cambio
    de la capa final, el formato que espera ``verificar_cifar100.py``.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    fc_before = model.fc.weight.detach().clone() if record is not None else None
    losses, labels_seen = [], Counter()
    if teacher is not None:
        teacher.eval()
    for xb, yb in batches:
        xb = preprocess(xb, augment=augment, generator=generator) if preprocess else xb.float()
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if teacher is not None and distill_weight > 0:
            with torch.no_grad():
                teacher_logits = teacher(xb)
            loss = loss + distill_weight * distillation_loss(logits, teacher_logits, old_classes)
        if not torch.isfinite(loss):
            raise RuntimeError("La pérdida de entrenamiento no es finita")
        loss.backward()
        optimizer.step()
        if record is not None:
            losses.append(loss.item())
            labels_seen.update(yb.detach().cpu().tolist())
    if record is not None:
        record.update({
            "optimizer_steps": len(losses), "samples_processed": sum(labels_seen.values()),
            "classes_seen": len(labels_seen), "label_counts": dict(labels_seen), "batch_losses": losses,
            "fc_weight_delta_l2": (model.fc.weight.detach() - fc_before.to(model.fc.weight.device)).norm().item(),
        })
    return model.state_dict()


@torch.no_grad()
def evaluate(model, x, y, device, preprocess: Optional[Callable] = None, batch_size: int = 256,
             group_a_classes: Optional[Sequence[int]] = None) -> dict:
    """Precisión y diagnósticos sobre tensores en memoria.

    Si se indican las clases del grupo A, también devuelve:

    * ``group_aware_accuracy``: precisión restringiendo el argmax al grupo
      verdadero de cada imagen (diagnóstico, no es la precisión de 100 clases);
    * ``pred_share_a`` / ``pred_share_b``: proporción de predicciones en cada grupo.
    """
    model.eval()
    total = len(y)
    if total == 0:
        raise RuntimeError("La evaluación no contiene muestras")
    correct, group_correct, loss_sum, pred_a = 0, 0, 0.0, 0
    predictions, targets = Counter(), Counter()
    mask_a = None
    for start in range(0, total, batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size].to(device)
        xb = (preprocess(xb) if preprocess else xb.float()).to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        predictions.update(pred.cpu().tolist())
        targets.update(yb.cpu().tolist())
        if group_a_classes is not None:
            if mask_a is None:
                mask_a = torch.zeros(logits.shape[1], dtype=torch.bool, device=device)
                mask_a[torch.as_tensor(list(group_a_classes), device=device)] = True
            in_a = mask_a[yb]
            masked = logits.masked_fill(~(mask_a.unsqueeze(0) == in_a.unsqueeze(1)), float("-inf"))
            group_correct += (masked.argmax(dim=1) == yb).sum().item()
            pred_a += mask_a[pred].sum().item()
    result = {"correct": correct, "total": total, "accuracy_percent": 100 * correct / total,
              "mean_cross_entropy": loss_sum / total,
              "prediction_counts": dict(predictions), "target_counts": dict(targets)}
    if group_a_classes is not None:
        result.update({"group_aware_accuracy_percent": 100 * group_correct / total,
                       "pred_share_a_percent": 100 * pred_a / total,
                       "pred_share_b_percent": 100 * (total - pred_a) / total})
    return result


def retention_percent(before: float, after: float) -> Optional[float]:
    """Porcentaje de la precisión inicial que se conserva (None si no había nada que conservar)."""
    return None if before <= 0 else 100 * after / before


def steps_per_epoch(n_samples: int, batch_size: int) -> int:
    return math.ceil(n_samples / batch_size)


# ------------------------------------------------------------------
# Adaptadores para scripts que usan DataLoader (DomainNet, MVTec)
# ------------------------------------------------------------------
def loader_batches(loader, epochs: int, max_steps: Optional[int] = None) -> Iterator:
    """Recorre un DataLoader ``epochs`` veces, cortando tras ``max_steps`` lotes si se indica."""
    steps = 0
    for _ in range(epochs):
        for batch in loader:
            yield batch
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return


@torch.no_grad()
def accuracy_on_loader(model, loader, device) -> float:
    """Precisión (fracción 0-1) sobre un DataLoader ya preprocesado."""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0
