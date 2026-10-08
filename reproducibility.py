"""Semillas, determinismo y registro del entorno de cada ejecución."""

import hashlib
import os
import platform
import random
import sys
from importlib import metadata

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch y pide algoritmos deterministas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Necesario para que cuBLAS sea determinista en GPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # warn_only: avisa en vez de fallar si alguna operación no tiene versión determinista.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _packages_digest():
    """SHA-256 de la lista ordenada de paquetes instalados y sus versiones."""
    packages = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions()
                      if dist.metadata.get("Name"))
    return hashlib.sha256("\n".join(packages).encode("utf-8")).hexdigest(), len(packages)


def environment_info() -> dict:
    """Datos del entorno que se guardan en config.json para poder repetir la ejecución."""
    digest, count = _packages_digest()
    return {
        "image_revision": os.environ.get("CFL_IMAGE_REVISION", "fuera de Docker"),
        "image_build_date": os.environ.get("CFL_IMAGE_BUILD_DATE"),
        "image_variant": os.environ.get("CFL_IMAGE_VARIANT"),
        "data_preloaded": os.environ.get("CFL_DATA_PRELOADED") == "1",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
        "packages_sha256": digest,
        "packages_count": count,
    }
