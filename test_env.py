"""
Script de verificación rápida del entorno y funcionamiento del proyecto.
Comprueba en pocos segundos que:
1. PyTorch y Torchvision funcionan correctamente.
2. Flower Datasets descarga y filtra particiones de CIFAR-100.
3. El modelo ResNet-18 modificado realiza forward y backward pass.
4. Matplotlib genera y guarda gráficos sin errores.
"""
import sys
import os
import time

print("=" * 60)
print(" VERIFICACION DEL ENTORNO DE APRENDIZAJE FEDERADO CONTINUO ")
print("=" * 60)

# 1. Imports
print("[1/4] Comprobando dependencias (PyTorch, Torchvision, Datasets)...")
start = time.time()
import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flwr_datasets import FederatedDataset
from reproducibility import seed_everything

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"      -> Dispositivo: {device} (hilos CPU: {torch.get_num_threads()})")
print(f"      -> Dependencias cargadas en {time.time() - start:.2f}s.")

# 2. Modelo
print("[2/4] Verificando arquitectura ResNet-18 para CIFAR-100...")
def get_model():
    model = resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

model = get_model().to(device)
dummy_x = torch.randn(4, 3, 32, 32).to(device)
dummy_y = torch.tensor([0, 1, 2, 3]).to(device)
out = model(dummy_x)
loss = nn.functional.cross_entropy(out, dummy_y)
loss.backward()
print(f"      -> Forward y backward pass correctos. Loss inicial: {loss.item():.4f}")

# 3. Dataset
print("[3/4] Verificando carga y particionado de CIFAR-100...")
start = time.time()
fds = FederatedDataset(
    dataset="uoft-cs/cifar100",
    revision="aadb3af77e9048adbea6b47c21a81e47dd092ae5",
    partitioners={"train": 10},
    seed=42,
)
p0 = fds.load_partition(0, "train")
print(f"      -> Partición de cliente cargada con {len(p0)} muestras en {time.time() - start:.2f}s.")

# 4. Matplotlib
print("[4/4] Verificando generación de gráficos con Matplotlib...")
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([0, 1, 2], [10, 25, 40], label="Test Loss")
ax.set_title("Test Chart")
test_chart = "test_verification_chart.png"
fig.savefig(test_chart)
plt.close(fig)
if os.path.exists(test_chart):
    os.remove(test_chart)
    print("      -> Gráfico generado y guardado correctamente.")

print("=" * 60)
print(" ¡EL ENTORNO ESTA 100% OPERATIVO Y LISTO PARA EJECUTAR! ")
print("=" * 60)
