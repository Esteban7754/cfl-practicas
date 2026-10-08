# Aprendizaje federado continuo

Proyecto de prácticas para estudiar el olvido catastrófico en modelos de clasificación de imágenes entrenados de forma federada y secuencial. Los experimentos comparan el entrenamiento sin memoria con diferentes capacidades de **replay buffer**, reutilizando muestras de la primera tarea al aprender la segunda.

> **Estado:** todavía no hay resultados válidos sobre el efecto del buffer. Consulta [ESTADO_EXPERIMENTAL.md](ESTADO_EXPERIMENTAL.md) antes de interpretar cualquier gráfico.

## Funcionamiento

Los scripts simulan clientes en un único proceso: cada cliente entrena una copia del modelo global y se agregan sus parámetros con FedAvg. Se usa una ResNet-18 adaptada a imágenes pequeñas, Adam y entropía cruzada. El código común está en `fl_core.py`.

1. Entrenar sobre el grupo o dominio A.
2. Continuar sobre B, incorporando muestras de A cuando se utiliza replay.
3. Evaluar la precisión en ambos grupos y medir cuánto se pierde en A.

La pérdida de precisión es `precisión de A antes de B − precisión de A después de B`, en puntos porcentuales (pp). Los porcentajes del buffer son la proporción de muestras de A que conserva cada cliente. Como esa métrica toca fondo cuando A cae a 0 %, `cifar100_three_buffer.py` registra también la retención (A después / A antes), la precisión de A restringida a sus clases, la proporción de predicciones B sobre el test de A y la precisión en las 100 clases.

## Estructura

| Archivo | Contenido |
| --- | --- |
| `cifar100_three_buffer.py` | Experimento principal con CIFAR-100: clases 0–49 (A) y 50–99 (B), buffers configurables, variantes de replay, auditoría. |
| `domainnet.py` | Cambio de dominio de fotos reales a bocetos con las 10 primeras clases de DomainNet. |
| `mvtecad_all_experiments.py` | Clasificación de las 15 categorías de MVTec AD (7 en A, 8 en B) y coste de almacenamiento del buffer. No es detección de anomalías. |
| `fl_core.py` | Modelo, FedAvg, entrenamiento local, evaluación, preprocesado y lotes de replay compartidos. |
| `experiment_common.py` | Argumentos, buffers anidados, tablas, CSV y gráficos de DomainNet y MVTec. |
| `cifar100_sampling.py`, `cifar100_validation.py` | Muestreo equilibrado y controles de aprendizaje. |
| `verificar_cifar100.py` | Recalcula las métricas desde los checkpoints con un evaluador independiente. |
| `agregar_semillas.py` | Media ± desviación típica entre semillas. |
| `comparar_replay.ps1`, `repetir_semillas.ps1` | Lanzadores de comparaciones completas y de varias semillas. |
| `legacy/` | Scripts históricos (`step_four.py`, `step_six_v1.py`, `three_buffer_size.py`, `three_buffer_v2.py`), conservados para reproducir los resultados antiguos. |
| `resultados/` | Una carpeta por ejecución; `resultados/historicos/` guarda los gráficos y registros antiguos. |
| `test_*.py` | Tests unitarios y una prueba de extremo a extremo con datos sintéticos (sin Internet). |

Los scripts de experimento solo entrenan al ejecutarse (`python script.py`); `cifar100_three_buffer.py` puede importarse sin efectos.

## Instalación

Se necesita **Docker Desktop** abierto. El Python nativo de Windows de este equipo no puede cargar `torch.dll` (WinError 4551, bloqueado por Control de aplicaciones), así que todo se ejecuta en el contenedor.

```powershell
docker build -t cfl-practicas .
```

El volumen de caché de Hugging Face (`cfl-hf-cache`) se crea solo la primera vez. Hay una variante opcional con GPU NVIDIA (`Dockerfile.gpu`, servicio `cfl-gpu` de Docker Compose) que todavía no se ha probado en este equipo.

> **Tras actualizar el repositorio, reconstruye la imagen** (`docker build -t cfl-practicas .`): ahora define `PYTHONPATH=/workspace`, necesario para los scripts de `legacy/`.

## Ejecución

- **Explorador o CMD:** doble clic en `run.bat` (o `ejecutar.bat`) para el menú, o `run.bat <script> [argumentos]`.
- **PowerShell:** `.\run.ps1` para el menú, o `.\run.ps1 <script> [argumentos]`. Ahora también admite argumentos y funciona desde cualquier carpeta.
- **VS Code / Cursor:** `Ctrl + Shift + B` ejecuta el archivo abierto en Docker; la tarea de test lanza los tests unitarios.
- **Docker Compose:** `docker compose run --rm cfl python cifar100_three_buffer.py --quick`.

```powershell
.\run.ps1 test                                   # verificación rápida del entorno (verificar_entorno.py)
.\run.ps1 unittest                               # tests unitarios
.\run.ps1 cifar100_three_buffer.py --quick --output-dir resultados/prueba
.\run.ps1 legacy/step_four.py                    # script histórico
```

Cada ejecución crea su propia carpeta en `resultados/` (o la indicada con `--output-dir`) con `config.json`, `results.csv`, `history.csv` y los gráficos; nunca sobrescribe los resultados anteriores.

### Opciones de `cifar100_three_buffer.py`

| Opción | Efecto |
| --- | --- |
| `--quick` / `--validation` | Prueba funcional mínima / piloto reducido con controles de aprendizaje. Sin ellas: datos completos. |
| `--buffers 0 5 10 ...` | Porcentajes a comparar (0–99). El 0 % es obligatorio y siempre se mide. |
| `--rounds N`, `--local-epochs N`, `--seed N` | Sobrescriben rondas, épocas locales (10 por defecto) y semilla (42). |
| `--controlled-replay` / `--no-controlled-replay` | Por defecto todas las variantes parten de los mismos pesos, usan buffers anidados y hacen las mismas actualizaciones. `--no-controlled-replay` recupera el diseño antiguo, en el que más buffer también implica más pasos. |
| `--replay-mix concat` / `balanced` | `concat` baraja B y el buffer juntos (diseño original). `balanced` reserva en cada lote una fracción fija para el buffer (`--replay-batch-fraction`, 0,5 por defecto). |
| `--distill-weight L` | Destilación sobre los logits de las clases A respecto al modelo de la fase A (estilo LwF). 0 la desactiva. |
| `--normalize` / `--no-normalize`, `--augment` / `--no-augment` | Normalización de CIFAR-100 y aumento de datos (recorte con relleno y volteo), activados por defecto. |
| `--phase1-checkpoint RUTA` | Reutiliza unos pesos de fase A. Exige su manifiesto (`.config.json`) y comprueba que coinciden los clientes, la semilla, los datos y el preprocesado. |
| `--audit` | Guarda diagnósticos y pesos para `verificar_cifar100.py`. |

Al entrenar la fase A, cada ejecución guarda `phase1_checkpoint.pt` y `phase1_checkpoint.config.json` para reutilizarlos más adelante.

`global_model_phase1.pt` (A = 26,88 %) se entrenó con `legacy/step_four.py`, 1 época local y **sin** normalización ni aumento. Para reutilizarlo hay que añadir `--no-normalize --no-augment`, o usar `.\comparar_replay.ps1 -ReuseCheckpoint`, que ya lo hace.

DomainNet y MVTec aceptan `--seed`, `--buffers`, `--rounds`, `--local-epochs`, `--output-dir` y `--equal-steps` / `--no-equal-steps` (mismas actualizaciones por variante, activado por defecto).

### Comparación completa verificada

```powershell
$env:OMP_NUM_THREADS = '4'; $env:MKL_NUM_THREADS = '4'
.\comparar_replay.ps1                                              # 0 % frente a 20 %, 5 épocas locales
.\comparar_replay.ps1 -Seeds 42,43,44 -Buffers 0,5,10,20            # varias semillas, con resumen
.\comparar_replay.ps1 -Seeds 42,43,44 -Buffers 0,20 -ReplayMix balanced -DistillWeight 1
.\comparar_replay.ps1 -ReuseCheckpoint -LocalEpochs 1               # repite el régimen antiguo
```

Entrena la fase A, compara los buffers con replay controlado y verifica las métricas desde los pesos (`verificar_cifar100.py`). Con varias semillas genera además `resumen_semillas.csv` y `perdida_A_media_semillas.png`.

### Pruebas reducidas y piloto

```powershell
.\run.ps1 cifar100_three_buffer.py --quick --audit --output-dir resultados/prueba_auditada
.\run.ps1 verificar_cifar100.py resultados/prueba_auditada
.\run.ps1 cifar100_three_buffer.py --validation --audit --buffers 0 20 --output-dir resultados/piloto
```

`--quick` usa 2 clientes, 1 ronda, 1 época y 100 muestras por grupo: comprueba el flujo, no el aprendizaje, y no genera gráficos comparativos. `--validation` usa 2 clientes, 500 imágenes por grupo y cliente, y cancela la comparación si el modelo no alcanza unos mínimos de aprendizaje en A, en B y en las propias imágenes del buffer. Esos umbrales son controles operativos, no una certificación científica. Una comparación inconcluyente hace que el verificador termine con código 2.

## Tests

```powershell
.\run.ps1 unittest
```

Cubren el muestreo, los argumentos, FedAvg, los lotes de replay, el preprocesado, las métricas y una ejecución completa de `cifar100_three_buffer.py` con datos sintéticos (unos 2 minutos en CPU). GitHub Actions los ejecuta en cada push (`.github/workflows/tests.yml`).

## Datos

- **CIFAR-100:** `uoft-cs/cifar100` mediante Flower Datasets, fijado al commit `aadb3af77e9048adbea6b47c21a81e47dd092ae5`.
- **DomainNet:** `wltjr1007/DomainNet` (splits `train` y `test`), fijado al commit `ee20570ae7a29c51571e55a9a17983f7625295d6`.
- **MVTec AD:** copia local en `data/mvtec_anomaly_detection/`, con una carpeta por categoría (`<categoría>/train/good/*.png`, `<categoría>/test/<tipo>/*.png`). Se obtiene desde el [formulario oficial](https://www.mvtec.com/research-teaching/datasets/mvtec-ad); licencia CC BY-NC-SA 4.0.

La primera ejecución de CIFAR-100 y DomainNet necesita Internet para descargar los datos en la caché.

## Resultados históricos

`resultados/historicos/` contiene los gráficos y registros anteriores a fijar la semilla y el entorno, sin modificar. Contienen errores documentados: en `group_a_accuracy_loss_all_buffers.png` la barra «No Replay» **no se midió**, y la versión antigua de `step_six_v1.py` etiquetaba siempre los buffers como «5 %» y «15 %» (el primer bloque de `results_step_six_macbook.txt` corresponde en realidad al 25 % y al 50 %). Detalles en [su README](resultados/historicos/README.md).

Los checkpoints (`*.pt`) no se versionan en git porque pesan unos 45 MB cada uno.
