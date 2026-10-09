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
| `cfl.py`, `run.ps1`, `run.bat` | Interfaz única (`cfl.py`, dentro del contenedor) y lanzadores de Windows que construyen y ejecutan la imagen. |
| `Dockerfile`, `docker-compose.yml`, `docker/` | Imagen reproducible (dependencias, datos y código) y su punto de entrada. |
| `legacy/` | Scripts históricos (`step_four.py`, `step_six_v1.py`, `three_buffer_size.py`, `three_buffer_v2.py`), conservados para reproducir los resultados antiguos. |
| `resultados/` | Una carpeta por ejecución; `resultados/historicos/` guarda los gráficos y registros antiguos. |
| `test_*.py` | Tests unitarios y una prueba de extremo a extremo con datos sintéticos (sin Internet). |

Los scripts de experimento solo entrenan al ejecutarse (`python script.py`); `cifar100_three_buffer.py` puede importarse sin efectos.

## Inicio rápido (Docker)

Solo hace falta **Docker Desktop** abierto. Todo lo demás (Python, PyTorch, dependencias fijadas, CIFAR-100 y el propio código) va dentro de una imagen, así que la ejecución es idéntica en cualquier equipo y no necesita Internet una vez construida.

```powershell
.\run.ps1                 # menú interactivo (o doble clic en run.bat)
.\run.ps1 info            # versión de la imagen, datos, hilos y GPU
.\run.ps1 tests           # tests unitarios dentro del contenedor
.\run.ps1 rapido          # prueba funcional con CIFAR-100 real + verificación
.\run.ps1 comparar        # comparación 0 % / 20 %, 5 épocas locales, verificada desde los pesos
```

La primera vez, el lanzador construye la imagen automáticamente: descarga las dependencias y CIFAR-100 y tarda unos minutos. Después solo se reconstruye lo que cambia.

### Comandos

| Comando | Qué hace |
| --- | --- |
| `info` | Muestra la revisión del código de la imagen, versiones, hilos, GPU y datos disponibles. |
| `tests` | Tests unitarios. |
| `rapido` | `cifar100_three_buffer.py --quick --audit` y verificación independiente. Comprueba el flujo, no el aprendizaje. |
| `comparar [opciones]` | Comparación de replay con todos los datos, verificada desde los pesos; con varias semillas añade el resumen media ± desviación. Opciones: `--seeds 42 43 44`, `--buffers 0 5 10 20`, `--local-epochs 5`, `--rounds 5`, `--replay-mix balanced`, `--replay-batch-fraction 0.5`, `--distill-weight 1`, `--conjunto`, `--reuse-checkpoint`, `--threads N`, `--output-dir`, `--dry-run`. |
| `semillas SCRIPT [opciones]` | Repite `cifar100_three_buffer.py`, `domainnet.py`, `mvtecad_all_experiments.py` o `legacy/three_buffer_v2.py` con `--seeds` y resume. El resto de argumentos se pasa al script. |
| `construir` | Reconstruye la imagen con el código actual. |
| `python ...`, `bash` | Cualquier comando dentro del contenedor. |

Ejemplos:

```powershell
.\run.ps1 comparar --seeds 42 43 44 --buffers 0 5 10 20
.\run.ps1 comparar --seeds 42 43 44 --buffers 0 20 --replay-mix balanced --distill-weight 1
.\run.ps1 semillas domainnet.py --seeds 42 43 44 --buffers 0 10 20
.\run.ps1 python cifar100_three_buffer.py --validation --audit --buffers 0 20
```

Los resultados aparecen siempre en una carpeta nueva dentro de `resultados/`.

### Código congelado o modo desarrollo

Por defecto se ejecuta el código **que está dentro de la imagen**: cada ejecución queda ligada a una versión exacta. `config.json` registra la revisión de git, la fecha de construcción, las versiones y una huella (SHA-256) de todos los paquetes instalados. Si cambias el código, el lanzador te avisa de que la imagen es de otra versión; `.\run.ps1 construir` la actualiza.

Para probar cambios sin reconstruir, añade `-dev` delante del comando: monta tu carpeta sobre el código de la imagen.

```powershell
.\run.ps1 -dev tests
.\run.ps1 -dev comparar --reuse-checkpoint --local-epochs 1      # necesita global_model_phase1.pt de tu carpeta
```

### GPU NVIDIA (opcional)

Si `run.ps1` detecta una GPU NVIDIA te lo indica. Para usarla, construye la variante GPU una vez; a partir de ahí se usa automáticamente (`-cpu` fuerza la de CPU):

```powershell
.\run.ps1 -gpu construir
.\run.ps1 info            # debe mostrar "cuda_available": true y el nombre de la GPU
```

Requiere el controlador NVIDIA actualizado y Docker Desktop con WSL 2; no se ha podido probar en este equipo. En GPU los resultados pueden diferir en los últimos decimales de los de CPU, por eso la verificación exacta desde los pesos solo se hace para ejecuciones en CPU.

### Sin el lanzador

```powershell
docker compose build cfl
docker compose run --rm cfl comparar --seeds 42 43
docker compose run --rm dev tests                     # modo desarrollo
docker compose --profile gpu run --rm gpu info        # GPU
```

Opciones de construcción (`--build-arg`): `TORCH_VARIANT=cpu|cuda|system` (`system` usa el PyTorch de la imagen base, por ejemplo `BASE_IMAGE=nvcr.io/nvidia/pytorch:...`) y `PRELOAD_DATA=0`, que no incluye CIFAR-100 y lo descarga en el volumen `cfl-datos` la primera vez.

El Python nativo de Windows de este equipo no puede cargar `torch.dll` (WinError 4551, bloqueado por Control de aplicaciones): usa siempre Docker.


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
| `--joint` | Añade la cota superior: entrenamiento conjunto A+B desde cero con el mismo presupuesto (2 × rondas, mismos pasos por ronda). Va a `joint_results.csv` y añade a `results.csv` la distancia a esa cota (`gap_all_vs_joint`, `intransigence_b`). En `cfl comparar` es `--conjunto`; no depende de la variante, así que basta con pedirlo en una. |

Cada fila de `results.csv` incluye además las métricas estándar de aprendizaje continuo para dos tareas (`avg_acc_tasks`, precisión media por tarea; `bwt_a`, *backward transfer*) y el mismo modelo evaluado con **Weight Aligning** (columnas `wa_*`): la capa final reescalada para que las clases B no tengan pesos más grandes que las A. Es una corrección posterior, no cambia el entrenamiento, y `verificar_cifar100.py` la recalcula por su cuenta.

Al entrenar la fase A, cada ejecución guarda `phase1_checkpoint.pt` y `phase1_checkpoint.config.json` para reutilizarlos más adelante.

`global_model_phase1.pt` (A = 26,88 %) se entrenó con `legacy/step_four.py`, 1 época local y **sin** normalización ni aumento. Para reutilizarlo hay que añadir `--no-normalize --no-augment`, o usar `.\comparar_replay.ps1 -ReuseCheckpoint`, que ya lo hace.

DomainNet y MVTec aceptan `--seed`, `--buffers`, `--rounds`, `--local-epochs`, `--output-dir` y `--equal-steps` / `--no-equal-steps` (mismas actualizaciones por variante, activado por defecto).

### Comparación completa verificada

```powershell
.\run.ps1 comparar                                                   # 0 % frente a 20 %, 5 épocas locales
.\run.ps1 comparar --seeds 42 43 44 --buffers 0 5 10 20              # varias semillas, con resumen
.\run.ps1 comparar --seeds 42 43 44 --buffers 0 20 --replay-mix balanced --distill-weight 1
.\run.ps1 comparar --seeds 42 43 44 --buffers 0 5 10 20 --conjunto   # con la cota superior
.\run.ps1 -dev comparar --reuse-checkpoint --local-epochs 1          # repite el régimen antiguo
```

`comparar_replay.ps1` y `repetir_semillas.ps1` siguen funcionando con sus parámetros antiguos (`-Seeds 42,43`, `-Buffers 0,20`...) y los traducen a estos comandos.

Entrena la fase A, compara los buffers con replay controlado y verifica las métricas desde los pesos (`verificar_cifar100.py`). Con varias semillas genera además `resumen_semillas.csv` y `perdida_A_media_semillas.png`.

### Plan completo en una GPU NVIDIA

```powershell
.\lanzar_gpu.ps1                 # comprueba la GPU, construye la imagen CUDA, pasa los tests y ejecuta el plan
.\lanzar_gpu.ps1 -Simular        # lista las tareas pendientes sin ejecutarlas
.\lanzar_gpu.ps1 -SoloPlan       # reanuda el plan tras un corte (sin reconstruir ni pasar los tests)
.\run.ps1 -dev plan planes/plan_prueba.json   # el mismo plan en miniatura (--quick), en CPU, en minutos
```

`planes/plan_gpu.json` define los experimentos por orden de prioridad: las tres variantes con 5 semillas (42-46), la ablación de la fracción del lote (natural, 0,25, 0,75), ER-ACE, la referencia conjunta con presupuesto ×2, el replay con calendario (0,75-0,5-0,25-0,25-0,25) y el reparto Dirichlet (α = 0,5 y 0,1) con mezcla normal y lotes equilibrados. `plan_experimentos.py` entrena la fase 1 una sola vez por semilla y reparto, lanza varias tareas a la vez en la GPU, se puede reanudar (salta lo terminado y aparta lo que quedó a medias) y un solo Ctrl+C lo detiene todo. Al terminar agrega las semillas y `informe_plan.py` escribe `resultados/plan_gpu/INFORME.md` con tablas (media ± desviación) y gráficas. El progreso queda en `resultados/plan_gpu/_registros/` y `estado.json`.

Cada fila de `results.csv` incluye además, sin coste de entrenamiento apreciable: WA completo, solo pesos y solo sesgo (`wa_*`, `wa_w_*`, `wa_b_*`), las logits medias de las clases A y B sobre el test de A (`mean_logit_*`) y cRT, la capa final reentrenada con el buffer y otras tantas imágenes de B (`crt_*`).

### Pruebas reducidas y piloto

```powershell
.\run.ps1 rapido
.\run.ps1 python cifar100_three_buffer.py --validation --audit --buffers 0 20 --output-dir resultados/piloto
.\run.ps1 python verificar_cifar100.py resultados/piloto
```

`--quick` usa 2 clientes, 1 ronda, 1 época y 100 muestras por grupo: comprueba el flujo, no el aprendizaje, y no genera gráficos comparativos. `--validation` usa 2 clientes, 500 imágenes por grupo y cliente, y cancela la comparación si el modelo no alcanza unos mínimos de aprendizaje en A, en B y en las propias imágenes del buffer. Esos umbrales son controles operativos, no una certificación científica. Una comparación inconcluyente hace que el verificador termine con código 2.

## Tests

```powershell
.\run.ps1 tests
```

Cubren el muestreo, los argumentos, FedAvg, los lotes de replay, el preprocesado, las métricas y una ejecución completa de `cifar100_three_buffer.py` con datos sintéticos (unos 2 minutos en CPU). GitHub Actions los ejecuta en cada push (`.github/workflows/tests.yml`) y además construye la imagen y ejecuta los tests y `rapido` dentro de ella sin red.

## Datos

- **CIFAR-100:** `uoft-cs/cifar100` mediante Flower Datasets, fijado al commit `aadb3af77e9048adbea6b47c21a81e47dd092ae5`.
- **DomainNet:** `wltjr1007/DomainNet` (splits `train` y `test`), fijado al commit `ee20570ae7a29c51571e55a9a17983f7625295d6`.
- **MVTec AD:** copia local en `data/mvtec_anomaly_detection/`, con una carpeta por categoría (`<categoría>/train/good/*.png`, `<categoría>/test/<tipo>/*.png`). Se obtiene desde el [formulario oficial](https://www.mvtec.com/research-teaching/datasets/mvtec-ad); licencia CC BY-NC-SA 4.0.

CIFAR-100 viene dentro de la imagen y se usa sin conexión. DomainNet se descarga la primera vez en el volumen `cfl-datos` y se conserva entre ejecuciones. MVTec se lee de `data/` en tu carpeta (montada en solo lectura).

## Resultados históricos

`resultados/historicos/` contiene los gráficos y registros anteriores a fijar la semilla y el entorno, sin modificar. Contienen errores documentados: en `group_a_accuracy_loss_all_buffers.png` la barra «No Replay» **no se midió**, y la versión antigua de `step_six_v1.py` etiquetaba siempre los buffers como «5 %» y «15 %» (el primer bloque de `results_step_six_macbook.txt` corresponde en realidad al 25 % y al 50 %). Detalles en [su README](resultados/historicos/README.md).

Los checkpoints (`*.pt`) no se versionan en git porque pesan unos 45 MB cada uno.
