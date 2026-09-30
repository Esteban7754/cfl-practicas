# Aprendizaje federado continuo

Proyecto de prácticas para estudiar el olvido catastrófico en modelos de clasificación de imágenes entrenados de forma federada y secuencial. Los experimentos comparan el entrenamiento sin memoria con diferentes capacidades de **replay buffer**, reutilizando muestras de la primera tarea al aprender la segunda.

## Funcionamiento

Los scripts simulan clientes en un único proceso: cada cliente entrena una copia del modelo global y se agregan sus parámetros mediante una media entre clientes. Utilizan ResNet-18 adaptada a imágenes pequeñas, optimización Adam y pérdida de entropía cruzada.

1. Entrenar sobre el grupo o dominio A.
2. Continuar sobre B, incorporando muestras de A cuando se utiliza replay.
3. Evaluar la precisión en ambos grupos y medir cuánto se pierde en A.

La pérdida de precisión se calcula como `precisión de A antes de B - precisión de A después de B`, en puntos porcentuales. Los porcentajes del buffer corresponden a la proporción de muestras de A conservadas por cada cliente.

## Experimentos disponibles

| Archivo | Contenido |
| --- | --- |
| `step_four.py` | Experimento inicial con CIFAR-100: clases 0–49 como grupo A y 50–99 como grupo B, sin replay. |
| `step_six_v1.py` | CIFAR-100 con buffers del 5 % y 15 %, y resumen en consola. |
| `three_buffer_size.py` | Comparación de buffers del 5 %, 15 % y 25 %; entrena la primera fase en cada experimento y utiliza una referencia sin replay fijada en el código. |
| `three_buffer_v2.py` | Reutiliza los pesos de la primera fase para comparar buffers del 5 %, 15 % y 25 %; parte de los valores de referencia sin replay están fijados en el código. |
| `cifar100_three_buffer.py` | Comparación de CIFAR-100 con 0 %, 5 %, 15 % y 25 % de replay, tablas y gráficos. |
| `domainnet.py` | Cambio de dominio de imágenes reales a bocetos con las primeras 10 clases de DomainNet y buffers del 0 %, 5 %, 15 % y 25 %. |
| `mvtecad_all_experiments.py` | Clasificación de las 15 categorías de MVTec AD, divididas en grupos de 7 y 8; compara los mismos buffers y estima su tamaño en disco y en tensores. |

En MVTec AD las etiquetas representan categorías de objetos o texturas; el experimento evalúa clasificación de categorías, no detección ni localización de anomalías.

## Instalación

Se necesita Python y las siguientes bibliotecas: PyTorch, torchvision, NumPy, Matplotlib, Flower Datasets, Hugging Face Datasets y Pillow. No hay versiones de dependencias fijadas en el repositorio.

Desde la raíz del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib flwr-datasets datasets pillow
```

Los scripts seleccionan automáticamente CUDA, MPS o CPU según disponibilidad. Para usar CUDA, la instalación de PyTorch debe ser compatible con la GPU y su entorno. El entrenamiento en CPU puede tardar considerablemente.

## Datos

- **CIFAR-100:** los scripts lo cargan mediante `FederatedDataset` desde `uoft-cs/cifar100`.
- **DomainNet:** `domainnet.py` carga los splits `train` y `test` de `wltjr1007/DomainNet` mediante Hugging Face Datasets.
- **MVTec AD:** requiere una copia local del dataset. Cambia `CONFIG["data_dir"]` en `mvtecad_all_experiments.py`; actualmente contiene una ruta de otro equipo.

Las primeras ejecuciones de CIFAR-100 y DomainNet necesitan acceso a Internet para obtener los datos y espacio para la caché. MVTec debe mantener esta estructura:

```text
mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/*.png
│   └── test/<good_o_tipo_de_defecto>/*.png
├── cable/
└── ... resto de categorías
```

## Ejecución

Cada archivo es un experimento independiente. Por ejemplo:

```powershell
python cifar100_three_buffer.py
python domainnet.py
python mvtecad_all_experiments.py
```

Para ejecutar las versiones iniciales:

```powershell
python step_four.py
python step_six_v1.py
python three_buffer_size.py
python three_buffer_v2.py
```

Los parámetros se editan directamente en el diccionario `CONFIG` de cada script: número de clientes, tamaño de lote, épocas locales, rondas, tasa de aprendizaje y dispositivo. Las capacidades del buffer se definen en las llamadas al experimento al final del archivo. No hay una interfaz de argumentos de línea de comandos.

Los scripts ejecutan el entrenamiento al cargarse, por lo que importarlos también inicia el experimento. Los que generan gráficos los guardan en el directorio de ejecución y abren una ventana con Matplotlib; una nueva ejecución puede sobrescribir los PNG con el mismo nombre.

## Resultados incluidos

- `results` y `step_six_v1_results.txt`: registros de ejecuciones anteriores de CIFAR-100.
- `group_a_accuracy_loss_5_15.png` y `group_a_accuracy_loss_all_buffers.png`: gráficos de comparación de CIFAR-100.
- `group_a_accuracy_loss_all_buffers_v3.png`: gráfico adicional conservado en el repositorio.
- `domainnet_accuracy_loss_5_15.png` y `domainnet_accuracy_loss_all_buffers.png`: gráficos del cambio de dominio.
- `mvtec_continual_loss_5_15.png` y `mvtec_continual_loss_all.png`: gráficos de MVTec AD.

![Comparación de buffers en CIFAR-100](group_a_accuracy_loss_all_buffers.png)

Los resultados guardados corresponden a ejecuciones anteriores y pueden diferir de una ejecución nueva. El repositorio conserva distintas versiones de los experimentos, sin un entorno de dependencias fijado ni una configuración común de semillas aleatorias. Para comparar resultados, registra el script, sus parámetros, las dependencias y el dispositivo utilizados; distingue las referencias fijadas en código de las mediciones obtenidas en esa ejecución.
