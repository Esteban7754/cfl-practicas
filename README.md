# Replay con presupuesto controlado en aprendizaje federado continuo

Estudio del olvido catastrófico en **aprendizaje federado continuo incremental por clases**: un modelo se entrena de forma federada (10 clientes, FedAvg) primero con las clases 0-49 de CIFAR-100 (tarea **A**) y después con las 50-99 (tarea **B**). Cada cliente guarda un pequeño **buffer** de imágenes de A para repasarlas (replay). El trabajo compara estrategias de replay con un **presupuesto controlado**: todas parten de los mismos pesos, usan memorias anidadas y hacen exactamente el mismo número de actualizaciones, de modo que las diferencias se deben a *cómo* se usa la memoria y no a cuánto se entrena.

Este README es la referencia única del proyecto, tanto para personas como para agentes: qué se ha hecho, qué ha salido, cómo reproducirlo y qué hay que tener en cuenta al tocar el código.

---

## 1. Estado actual (13 de octubre de 2026)

**El estudio principal está terminado.** El plan completo (`planes/plan_gpu.json`, 56 tareas, 3-5 semillas por configuración) se ejecutó en una NVIDIA RTX 5050. Los resultados están en `resultados/plan_gpu/INFORME.md`, carpeta local que no se versiona (ver §7). Existe un borrador de artículo en un documento de Claude (privado) que recoge todo lo de esta sección.

### Resultados principales (CIFAR-100, reparto IID, buffer del 20 %)

Precisión en % (media ± desviación típica entre semillas). Antes de aprender B, A está en 48,0 ± 0,9 %.

| Método | Semillas | A final | B final | Retención de A | 100 clases | 100 clases + cRT |
|---|---|---|---|---|---|---|
| Sin replay (buffer 0 %) | 5 | 0,0 | 57,5 ± 1,4 | 0 % | 28,8 ± 0,7 | — |
| Mezcla normal (`concat`) | 5 | 26,1 ± 0,7 | 56,5 ± 1,1 | 54,3 % | 41,3 ± 0,8 | 46,1 ± 1,1 |
| Lotes equilibrados + LwF | 5 | 36,8 ± 1,1 | 50,9 ± 0,9 | 76,6 % | 43,8 ± 0,9 | 45,9 ± 0,8 |
| Calendario adaptativo | 3 | 34,1 ± 0,4 | 57,2 ± 0,1 | 70,1 % | 45,7 ± 0,2 | **49,2 ± 0,3** |
| Lotes equilibrados (f = 0,5) | 5 | 39,5 ± 0,3 | 52,1 ± 1,0 | 82,3 % | 45,8 ± 0,7 | 47,4 ± 1,0 |
| ER-ACE adaptado | 3 | **44,1 ± 0,3** | 50,0 ± 0,4 | **90,5 %** | **47,1 ± 0,1** | 47,9 ± 0,2 |
| Referencia conjunta ×1 (A+B a la vez, mismo presupuesto) | 5 | 44,9 | 46,2 | — | 45,6 ± 0,7 | — |
| Referencia conjunta ×2 (doble presupuesto) | 3 | 59,2 | 61,1 | — | 60,1 ± 0,6 | — |

### Conclusiones (comparaciones pareadas por semilla)

1. **Lo que decide es la proporción de memoria en cada lote.** Los lotes equilibrados superan a la mezcla normal en +4,5 puntos (100 clases) con cualquier buffer (5 de 5 semillas, p < 0,001). Con la fracción `natural` (misma proporción que la mezcla normal) la ventaja desaparece (+0,2, no significativa). La retención crece con la fracción y B baja; el total es máximo en torno a 0,5.
2. **ER-ACE adaptado es el mejor método sin correcciones,** y su ventaja crece cuanto menor es la memoria: +0,8 / +2,9 / +4,7 puntos sobre los lotes equilibrados con buffers del 20 / 10 / 5 % (3 de 3 semillas). Su memoria **debe** incluir ejemplos de la tarea actual; sin ellos B = 0 % (ver §5.3).
3. **cRT mejora todas las variantes.** La mejor combinación es calendario + cRT (49,2 %). **WA** solo ayuda si el modelo está sesgado hacia B (mezcla normal, fracciones bajas, calendario) y perjudica al resto: las logits de B sobre imágenes de A son negativas, y γ > 1 las hace más negativas.
4. **LwF empeora** los lotes equilibrados (−2,0 puntos; 5 de 5 semillas) con λ = 1 y T = 2, sin ajustar.
5. **Con datos desiguales (Dirichlet α = 0,5 y 0,1)** los lotes equilibrados siguen ganando (+2,2 a +4,5 puntos), con más coste en B. Con Dirichlet la retención puede pasar del 100 %, así que se comparan precisiones absolutas.
6. **Reproducibilidad:** en la misma GPU los resultados son idénticos. Entre RTX 5050 y 2 × T4 (Kaggle), con la semilla 42 y 12 configuraciones, la diferencia media es de 0,45 puntos (máx. 0,86), menor que la variación entre semillas.

### En curso: protocolo de 5 tareas (25 de octubre de 2026)

Tras revisar el estudio con el profesor se pasa a **5 tareas de 20 clases** con hiperparámetros fijos y solo las 5 configuraciones más prometedoras: mezcla normal, lotes equilibrados, calendario, ER-ACE y lotes equilibrados con Dirichlet α = 0,5. La memoria de cada cliente tiene un tamaño fijo (10 o 20 % de una tarea, es decir 1 000 o 2 000 imágenes en total) repartido entre las clases vistas, y la fracción de memoria por lote es (t−1)/t.

- Diseño y fases: [docs/plan_cinco_tareas.md](docs/plan_cinco_tareas.md). Variables y métricas: [docs/variables.md](docs/variables.md).
- Código: `cfl_tareas.py` (experimento), `memoria_cfl.py` (memoria por clases), `datasets_cfl.py` (datasets), `informe_tareas.py` (informe). El estudio de 2 tareas (`cifar100_three_buffer.py`) no cambia.
- Estado: código terminado y probado con `planes/plan_prueba_tareas.json`. Falta el piloto en GPU (semilla 42, `equilibrados`) y el plan completo `planes/plan_tareas.json` (21 tareas, ~1 día de GPU, estimación sin medir). Si todo va bien, después se probará con Tiny-ImageNet.

### Después: extensión B

Calibrar el clasificador **en el servidor, sin buffer**, con estadísticas por clase de los clientes (en la línea de CCVR, aplicado a tareas secuenciales). Plan por pasos con punto de decisión temprano en [docs/extension_calibracion_servidor.md](docs/extension_calibracion_servidor.md). Línea base a batir: calendario + cRT (49,2 %); margen hasta la referencia ×2 (60,1 %).

> `ESTADO_EXPERIMENTAL.md` describe el estado del 8 de octubre de 2026 (antes de estos resultados) y se conserva como registro histórico.

---

## 2. Guía rápida para agentes

- **Ejecuta siempre en Docker.** En el equipo del autor (Windows), el Python nativo no puede cargar `torch.dll` (bloqueado por Control de aplicaciones) y no hay `python` en el PATH. Todo comando de Python va por `.\run.ps1 ...` (o `.\run.ps1 -dev ...` para usar el código de la carpeta sin reconstruir la imagen).
- **Tests antes de cambiar nada que importe:** `.\run.ps1 -cpu -dev tests` (83 tests, unos 10 min en CPU; `-dev` no está disponible con la imagen GPU). Solo el protocolo de 5 tareas: `.\run.ps1 -cpu -dev tests test_memoria test_tareas test_plan`.
- **Protocolo de 5 tareas (`cfl_tareas.py`):** los hiperparámetros están en `FIJOS` y no se exponen por línea de órdenes; no los añadas como opciones sin decidirlo con el autor. La memoria debe seguir siendo de tamaño fijo, anidada entre buffers y sin recuperar imágenes descartadas (`test_memoria.py` lo comprueba).
- **Invariantes del protocolo que no se deben romper sin decidirlo explícitamente:**
  - `--controlled-replay` (por defecto): mismos pesos de partida, buffers anidados y equilibrados por clase, mismos pasos por ronda para cualquier buffer.
  - `--buffers` **debe incluir el 0** (es la referencia de cada ejecución).
  - La fase 1 se entrena una vez por semilla y reparto y se reutiliza con `--phase1-checkpoint`; el manifiesto `.config.json` valida clientes, semilla, datos, preprocesado y reparto.
  - `--loss ace` exige `--replay-mix balanced`.
- **Dónde está cada cosa:**
  - experimento: `cifar100_three_buffer.py`;
  - núcleo: `fl_core.py`, con modelo, FedAvg, `train_local`, `ace_loss`, `mixed_batches`, `weight_align` y `evaluate`;
  - orquestación: `plan_experimentos.py` + `planes/*.json`;
  - informe: `informe_plan.py`;
  - CLI: `cfl.py`.
- **Salidas:** siempre en una carpeta nueva de `resultados/`. Un plan escribe en `resultados/<nombre>/` con `estado.json`, `_registros/<tarea>.log` e `INFORME.md`.
- **Los scripts `.ps1` se guardan en UTF-8 con BOM y CRLF:** Windows PowerShell 5.1 los lee mal sin BOM.
- **El repositorio está en OneDrive.** Docker no puede leer sus *reparse points*, así que `run.ps1` copia los archivos de git a `%LOCALAPPDATA%\cfl-practicas\contexto-docker` antes de construir (`CFL_CONTEXTO`). Un archivo nuevo debe estar en git (o al menos añadido) para entrar en la imagen.
- **Commits y pushes:** el autor aprueba cada push. Ramas: `main` (principal) y `dev` (espejo).

---

## 3. Requisitos

- **Docker Desktop** con WSL 2. Es lo único imprescindible: Python, PyTorch, dependencias fijadas, CIFAR-100 y el código van dentro de la imagen.
- **GPU NVIDIA (opcional, recomendada para el plan completo):** controlador reciente en Windows. Se ha usado una RTX 5050 de 8 GB (arquitectura Blackwell) con el controlador 617.42; la imagen GPU instala PyTorch 2.14.1 + CUDA 13.0.
- Sin GPU todo funciona en CPU, pero el plan completo tardaría días.

---

## 4. Cómo reproducir

### 4.1 Comprobación rápida (CPU, minutos)

```powershell
.\run.ps1 info                                # versión de la imagen, datos, GPU
.\run.ps1 tests                               # tests unitarios
.\run.ps1 rapido                              # flujo completo con CIFAR-100 real (--quick) + verificación
.\run.ps1 -dev plan planes/plan_prueba.json   # el plan completo en miniatura (13 tareas, --quick)
```

### 4.2 Plan completo en GPU (los resultados de la §1)

```powershell
.\lanzar_gpu.ps1                      # 1.ª vez: comprueba la GPU, construye la imagen CUDA, pasa los tests y ejecuta el plan
.\vigilar_gpu.ps1 -Paralelo 2         # recomendado: el mismo plan con recuperación automática de cuelgues
.\vigilar_gpu.ps1 -Paralelo 2 -Hasta 00:00   # por tandas: no empieza tareas nuevas a partir de esa hora
.\lanzar_gpu.ps1 -Simular             # lista lo pendiente sin ejecutar
```

- **Duración:** unas 22 h de GPU en una RTX 5050 con 2 tareas a la vez. Por tipo de tarea: fase 1 unos 22 min (3 a la vez); 3 buffers unos 50-58 min; 4 buffers (ER-ACE) unos 70 min; referencia ×2 unos 84 min; Dirichlet α = 0,1 hasta 63 min.
- **Reanudable:** relanzar el mismo comando salta lo terminado. Las tareas a medias se apartan (`*_incompleta_<fecha>`) y se repiten desde el principio. Un solo Ctrl+C lo detiene todo.
- **Límite de tiempo:** `-Hasta HH:MM` u `-Horas N`. Las tareas en curso a esa hora terminan antes de parar, y el código de salida 3 significa pausa.
- Mientras corre, el PC no se suspende. Al terminar o pausar **sí puede suspenderse**: si se relanza desde otra sesión remota, el comando no se ejecuta hasta que el PC despierte.

### 4.3 En Kaggle (2 × T4)

`kaggle_plan.ipynb` clona el repositorio en un commit fijo (`COMMIT = "6ceab93"`), ejecuta `planes/plan_prueba.json` y después `planes/plan_kaggle.json` (el plan con la semilla 42) y empaqueta `resultados_kaggle.tar.gz` sin los `.pt`. **Ese commit es anterior a la corrección de ER-ACE (67b2740):** actualiza `COMMIT` antes de reutilizarlo. Sesión con GPU T4 ×2 e Internet activado; el plan con la semilla 42 tarda unas 6 h 40 min.

**Protocolo de 5 tareas en Kaggle:** `kaggle_tareas.ipynb` clona el commit `COMMIT`, pasa los tests y el plan en miniatura y ejecuta `planes/plan_tareas.json` **semilla a semilla** (`SEMILLAS`, por defecto 42 y 43), sin empezar tareas nuevas a partir de `HORAS` (9,5 h desde el inicio). Deja `resultados_kaggle_tareas.tar.gz` sin los `.pt`. Las semillas que falten van en otra sesión; los paquetes se combinan descomprimiéndolos en la misma carpeta y ejecutando `informe_tareas.py planes/plan_tareas.json`.

### 4.4 Protocolo de 5 tareas

```powershell
.\run.ps1 -cpu -dev plan planes/plan_prueba_tareas.json   # miniatura (--quick), unos 20 min en CPU
.\run.ps1 -gpu construir                                 # la imagen GPU no lee la carpeta: reconstruir tras cada commit
.\vigilar_gpu.ps1 -Plan planes/plan_tareas.json -Paralelo 2   # plan completo con recuperación de cuelgues
.\run.ps1 -cpu -dev python cfl_tareas.py --quick --metodo balanced --joint   # un experimento suelto
```

El informe queda en `resultados/<plan>/INFORME.md`, con la curva incremental y la matriz R en `informe/`.

### 4.5 Un experimento suelto

```powershell
.\run.ps1 python cifar100_three_buffer.py --buffers 0 10 20 --replay-mix balanced --loss ace --seed 42 --local-epochs 5 --rounds 5
.\run.ps1 comparar --seeds 42 43 44 --buffers 0 5 10 20 --replay-mix balanced   # varias semillas + verificación + resumen
```

---

## 5. Protocolo experimental

### 5.1 Configuración

| Elemento | Valor |
|---|---|
| Datos | CIFAR-100, `uoft-cs/cifar100` vía Flower Datasets, revisión `aadb3af77e9048adbea6b47c21a81e47dd092ae5`; 25 000 de entrenamiento y 5 000 de test por tarea |
| Clientes | 10; IID (2 500 imágenes por tarea y cliente) o Dirichlet por clase (`flwr` `DirichletPartitioner`, α = 0,5 / 0,1) |
| FedAvg | Media simple en IID; ponderada por tamaño en Dirichlet (los clientes sin datos no participan) |
| Modelo | ResNet-18 de torchvision con primera convolución 3 × 3 y sin max-pooling, 100 salidas |
| Entrenamiento | Adam, lr 1e-3, lote 32, 5 épocas locales (el valor por defecto del script es 10: el plan pasa `--local-epochs 5`), 5 rondas por fase |
| Preprocesado | Normalización con media y desviación de CIFAR-100; recorte aleatorio con relleno de 4 y volteo horizontal |
| Buffers | 0, 5, 10 y 20 % de las imágenes de A de cada cliente, anidados y equilibrados por clase |
| Semillas | 42-46 para mezcla normal, lotes equilibrados y LwF; 42-44 para el resto |

**Presupuesto controlado:** en cada ronda de la fase 2, cada cliente hace `épocas · ceil(n_B / 32)` pasos, sea cual sea el buffer.

### 5.2 Variantes y sus opciones

| Variante | Opciones de `cifar100_three_buffer.py` |
|---|---|
| Mezcla normal | `--replay-mix concat` (por defecto): B y el buffer barajados juntos, truncado al presupuesto |
| Lotes equilibrados | `--replay-mix balanced --replay-batch-fraction 0.5`: `round(32·f)` muestras del buffer (con reposición) en cada lote |
| Ablación de la fracción | `--replay-batch-fraction natural / 0.25 / 0.75`; `natural` = tamaño del buffer / (tamaño de B + tamaño del buffer) |
| Calendario adaptativo | `--replay-fraction-schedule 0.75,0.5,0.25,0.25,0.25` (una fracción por ronda) |
| LwF | `--distill-weight 1` (KL sobre las logits de A respecto al modelo de la fase 1, T = 2) |
| ER-ACE adaptado | `--replay-mix balanced --loss ace` |
| Referencia conjunta | `--joint [--joint-budget-factor 2]`: A y B a la vez desde cero; va a `joint_results.csv` |
| Datos desiguales | `--partition dirichlet --dirichlet-alpha 0.5` |
| Solo fase 1 / reutilizarla | `--only-phase1`, `--phase1-checkpoint RUTA` |
| Correcciones posteriores | WA (completo, solo pesos, solo sesgo) siempre; cRT con `--crt-epochs 2` (por defecto; 0 lo desactiva) |

### 5.3 ER-ACE: detalle importante

`fl_core.ace_loss` enmascara las clases antiguas en las **muestras entrantes** y usa la entropía cruzada completa en las de la **memoria**. La máscara va por el **origen** de la muestra, con un tercer elemento del lote (`mixed_batches(..., with_origin=True)`), no por su clase. En el método original la memoria contiene también la tarea actual. Por eso `Experiment.ace_memory` (en `cifar100_three_buffer.py`) construye, para cada cliente, el buffer de A **más el mismo número de imágenes de B**, equilibradas por clase (ya están en el cliente: no aumenta la memoria de A). Con buffer 0 % no hay memoria y se usa entropía cruzada normal. La primera versión, con la memoria solo de A, daba B = 0 % en todos los buffers.

### 5.4 Métricas (`results.csv`, una fila por buffer)

| Columna | Significado |
|---|---|
| `step3_acc_a` | A antes de la fase 2 |
| `step4_acc_a`, `step4_acc_b` | A y B finales, eligiendo entre las 100 clases |
| `acc_all_100_classes` | Test completo; métrica principal (igual a la media de A y B) |
| `retention_a_percent` | A final / A antes × 100 |
| `bwt_a` | A final − A antes |
| `avg_acc_tasks` | Media de las precisiones por tarea |
| `group_aware_acc_a` | A restringida a sus propias clases (diagnóstico) |
| `pred_share_b_on_test_a` | Proporción de predicciones B sobre el test de A |
| `mean_logit_a_test_a`, `mean_logit_b_test_a` | Logits medias de las clases A y B sobre imágenes de A |
| `wa_*`, `wa_w_*`, `wa_b_*` | El mismo modelo tras WA completo, solo pesos y solo sesgo (`wa_gamma` = γ) |
| `crt_*` | Tras cRT (solo con buffer > 0) |

`joint_results.csv` guarda la referencia conjunta (`acc_a`, `acc_b`, `acc_all_100_classes`, `budget_factor`). Con varias semillas, `agregar_semillas.py` genera `resumen_semillas.csv` y `resumen_conjunto.csv`.

---

## 6. Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `cifar100_three_buffer.py` | Experimento principal (fases, buffers, variantes, correcciones, referencia conjunta, auditoría) |
| `fl_core.py` | Modelo, FedAvg, entrenamiento local, pérdidas (LwF, ACE), lotes de replay, WA, evaluación y preprocesado |
| `cfl_tareas.py`, `memoria_cfl.py`, `datasets_cfl.py` | Protocolo de 5 tareas: experimento, memoria fija por clases y especificación de datasets |
| `informe_tareas.py` | `INFORME.md` y gráficas de un plan de 5 tareas |
| `cifar100_sampling.py`, `cifar100_validation.py` | Muestreo equilibrado por clase y contrato de validación del punto de control de la fase 1 |
| `plan_experimentos.py` | Orquestador de planes: fase 1 compartida, tareas en paralelo, reparto entre GPUs, reanudación, límite de tiempo |
| `informe_plan.py` | Genera `INFORME.md` y las gráficas de un plan |
| `planes/plan_gpu.json` | Plan completo (56 tareas). `plan_kaggle.json`: solo la semilla 42. `plan_prueba.json`: miniatura con `--quick` |
| `cfl.py` | CLI dentro del contenedor: `info`, `entorno`, `tests`, `rapido`, `comparar`, `semillas`, `plan` |
| `run.ps1`, `run.bat` | Lanzador de Windows: construye y ejecuta la imagen (`-dev`, `-gpu`, `-cpu`) |
| `lanzar_gpu.ps1` | Prepara la GPU (comprobación, imagen CUDA, tests) y ejecuta un plan; `-SoloPlan`, `-Hasta`, `-Horas`, `-Paralelo`, `-Simular` |
| `vigilar_gpu.ps1` | Ejecuta `lanzar_gpu.ps1 -SoloPlan` y recupera los cuelgues de la GPU (§8) |
| `Dockerfile`, `docker-compose.yml`, `docker/` | Imagen reproducible (CPU o CUDA, `TORCH_VARIANT`) con CIFAR-100 incluido; `entrypoint.sh` |
| `verificar_cifar100.py` | Recalcula las métricas desde los pesos con un evaluador independiente (solo exacto en CPU) |
| `agregar_semillas.py` | Media ± desviación entre semillas |
| `kaggle_tareas.ipynb` | Cuaderno de Kaggle del protocolo de 5 tareas (semilla a semilla, con límite de horas) |
| `kaggle_plan.ipynb`, `kaggle_experimento.ipynb` | Cuadernos de Kaggle del estudio de 2 tareas (plan con la semilla 42; experimento suelto) |
| `docs/extension_calibracion_servidor.md` | Diseño de la siguiente extensión |
| `domainnet.py`, `mvtecad_all_experiments.py`, `experiment_common.py` | Experimentos secundarios (cambio de dominio, MVTec AD como clasificación) |
| `legacy/`, `comparar_replay.ps1`, `repetir_semillas.ps1` | Scripts históricos y lanzadores antiguos, conservados por compatibilidad |
| `test_*.py` | Tests unitarios y de extremo a extremo con datos sintéticos, sin red. GitHub Actions los ejecuta en cada push |
| `resultados/` | Una carpeta por ejecución. En git solo están las ejecuciones antiguas (auditoría del 2 de octubre e `historicos/`) |

---

## 7. Resultados y datos

- **En git están los resultados esenciales** de los dos planes: `resultados/plan_gpu/` (RTX 5050, todas las semillas) y `resultados/kaggle_plan_2026-10-10/resultados/plan_kaggle/` (2 × T4, semilla 42, con ER-ACE **sin** corregir). En concreto:
  - `INFORME.md`, `informe/*.png` y `estado.json`;
  - los resúmenes `resumen_*.csv`;
  - por semilla, `results.csv`, `joint_results.csv`, `config.json` e `history.csv`;
  - los manifiestos de la fase 1.

  Con eso se pueden leer y reanalizar todos los números sin volver a entrenar. **No están en git** los `audit.json`, los registros ni los puntos de control `*.pt` (unos 45 MB cada uno, ignorados por `.gitignore`). Para obtenerlos hay que relanzar el plan.
- **Estructura de un plan terminado:**
  - `resultados/plan_gpu/<trabajo>/seed_<n>/` con `results.csv`, `joint_results.csv` (si procede), `config.json`, `audit.json` e `history.csv`;
  - `_fase1/<reparto>_seed<n>/phase1_checkpoint.pt`;
  - `_registros/<tarea>.log`, más `vigilante.log` y `diagnostico_*.txt` si hubo cuelgues;
  - `estado.json`;
  - `INFORME.md` e `informe/*.png`.
- **CIFAR-100** viene dentro de la imagen y funciona sin conexión. DomainNet se descarga en el volumen `cfl-datos`. MVTec AD se lee de `data/` (licencia CC BY-NC-SA 4.0, formulario oficial).
- `resultados/historicos/` contiene resultados anteriores a fijar semillas y entorno, con errores documentados en [su README](resultados/historicos/README.md). No son comparables con los actuales.

---

## 8. Problemas conocidos y soluciones

| Problema | Síntoma | Solución |
|---|---|---|
| **La GPU se cuelga dentro de WSL** (desde la actualización de Windows 26H2 del 12 de octubre de 2026) | Las tareas siguen vivas pero no avanzan, la GPU está al 0-4 % y el controlador registra `nvlddmkm` 153 ("Error occurred on GPUID: 100"), a veces unos minutos *después* de que se paren | Usar `vigilar_gpu.ps1`: si el registro de **cualquier** tarea en curso pasa 10 min sin escribirse, guarda un diagnóstico, reinicia WSL y Docker y relanza el plan. Con 2 tareas en paralelo (`-Paralelo 2`) los cuelgues son menos frecuentes. Causa no identificada; no se ha investigado más |
| Windows Update cambia el controlador de NVIDIA | Controlador antiguo (610.88) tras reiniciar | Reinstalar el controlador desde nvidia.com (617.42 o posterior) |
| Docker Desktop no arranca tras matarlo | `initializing Secrets Engine ... engine.sock ... El sistema no tiene acceso al archivo` | Cerrar **todos** los procesos `Docker Desktop` y `com.docker.*`, `wsl --shutdown`, borrar `%LOCALAPPDATA%\docker-secrets-engine\engine.sock*` y abrir Docker de nuevo (`vigilar_gpu.ps1` ya lo hace) |
| Docker recién arrancado | La primera tarea muere con `CUDA error: unknown error` o `unexpected EOF` | Esperar 2-3 min tras abrir Docker Desktop antes de lanzar el plan |
| Construcción de la imagen desde OneDrive | `invalid file request` | Ya resuelto: `run.ps1` construye desde una copia fuera de OneDrive |
| Jugar o usar otras aplicaciones gráficas mientras corre el plan | Cuelgue al abrir el juego | Evitarlo, o pausar el plan entre tareas |
| Aviso "La imagen tiene el código X y tu carpeta está en Y" | — | Inofensivo si los commits nuevos no tocan los experimentos; si los tocan, `.\run.ps1 -gpu construir` |
| Mensaje de Docker "Debug this Compose error with Gordon" | Al pausar (código 3) o interrumpir | Inofensivo |

---

## 9. Historial breve

| Fecha | Hito |
|---|---|
| 2 de octubre | Auditoría del protocolo antiguo: todas las variantes acababan con A = 0 % |
| 8 de octubre | Núcleo común `fl_core.py`, normalización y aumento, replay controlado, lotes equilibrados, LwF, imagen Docker |
| 8-9 de octubre | Primeros resultados con 1 y 3 semillas (CPU y Kaggle) |
| 9-10 de octubre | Plan automático, WA y variantes, cRT, ER-ACE, Dirichlet, calendario, referencia ×2; plan con la semilla 42 en Kaggle (2 × T4) |
| 10 de octubre | Corrección de ER-ACE (memoria con A y B; máscara por origen) |
| 11-13 de octubre | Plan completo en la RTX 5050 (56 tareas); vigilante de cuelgues |
| 13 de octubre | Borrador del artículo con todos los resultados |
| 25 de octubre | Revisión con el profesor: protocolo de 5 tareas de 20 clases, hiperparámetros fijos, 5 configuraciones; código y plan listos |
