# Plan de implementación: 5 tareas de 20 clases

Decisiones acordadas con el profesor (25 de octubre de 2026) y aprobadas sobre la propuesta:

- 5 tareas de 20 clases (T1 = 0-19, …, T5 = 80-99), orden fijo para todas las semillas.
- Memoria de tamaño fijo por cliente, repartida a partes iguales entre las clases vistas.
- Fracción de replay por lote proporcional: clases antiguas / clases vistas.
- Hiperparámetros fijos; solo 5 configuraciones de replay; 3 semillas (42, 43, 44).
- Documento de variables; otro dataset (Tiny-ImageNet) al final si todo va bien.

---

## 0. Decisión de arquitectura: script nuevo

Se crea `cfl_tareas.py` en lugar de modificar `cifar100_three_buffer.py`.

- El script de 2 tareas queda intacto: reproduce los resultados ya publicados (§1 del README) y sus 58 tests siguen valiendo.
- `fl_core.py` ya es casi genérico: `weight_align`, `ace_loss`, `mixed_batches`, `train_local` y `get_model(num_classes)` reciben las clases como parámetro. Solo `evaluate` necesita generalizarse (ver fase 2).
- Se reutilizan sin cambios: `cifar100_sampling.balanced_order`, `reproducibility`, el mecanismo de manifiesto de la fase 1 y el orquestador de planes.

---

## 1. Protocolo nuevo

### 1.1 Tareas y presupuesto

| Elemento | Valor |
|---|---|
| Tareas | 5 × 20 clases, orden 0-99 (las etiquetas de CIFAR-100 son alfabéticas) |
| Datos por tarea | 10 000 de entrenamiento (1 000 por cliente en IID), 2 000 de test |
| Rondas | 5 por tarea → 25 en total |
| Pasos por ronda | `5 épocas · ceil(n_tarea / 32)` por cliente, igual con cualquier buffer (presupuesto controlado) |
| Fase compartida | Solo T1, una vez por semilla y reparto (`--only-phase1` / `--phase1-checkpoint`) |

### 1.2 Memoria (decisión 1)

- **Tamaño fijo por cliente**: `M = p % · imágenes de una tarea del cliente`. Con p = 10 / 20 % en IID son 100 / 200 imágenes por cliente, es decir **1 000 / 2 000 en total**. 2 000 es exactamente el presupuesto estándar de iCaRL en CIFAR-100, lo que facilita comparar con la literatura.
- **Reparto**: `M / clases vistas` por clase (entre las clases que el cliente tiene; en Dirichlet, las que tenga). Tras T1 son 10 por clase y cliente con 20 %; tras T4, 2-3.
- **Actualización al terminar cada tarea**: se recorta cada clase antigua a su nueva cuota y se añaden las imágenes de la tarea recién aprendida. El cliente **no** conserva los datos completos de tareas pasadas: solo la memoria.
- **Anidada**: el orden de cada clase se fija una vez con `balanced_order(seed + cliente)` y se toman prefijos. Así la memoria del 10 % está siempre contenida en la del 20 %, también tras los recortes.
- Buffers: **0, 10 y 20 %**.

### 1.3 Fracción por lote (decisión 2)

- Lotes equilibrados y ER-ACE: `f_t = (t − 1) / t` en la tarea t (1/2, 2/3, 3/4, 4/5), `round(32 · f_t)` muestras de memoria por lote.
- Calendario: `0,75 → 0,5 → 0,25 → 0,25 → 0,25` dentro de **cada** tarea (fijo, no proporcional).

### 1.4 Las 5 configuraciones

| Id | Método | Opciones |
|---|---|---|
| `normal` | Mezcla normal (línea base) | `--metodo concat` |
| `equilibrados` | Lotes equilibrados, f proporcional | `--metodo balanced` |
| `calendario` | Calendario adaptativo | `--metodo calendario` |
| `er_ace` | ER-ACE adaptado | `--metodo ace` |
| `dir05_equilibrados` | Lotes equilibrados con Dirichlet α = 0,5 | `--metodo balanced --partition dirichlet` |

Más, en cada semilla: buffer 0 % (sin replay) dentro de cada trabajo y la **referencia conjunta** (100 clases desde cero, 25 rondas con los pasos de una ronda de tarea) en `equilibrados`.

Adaptación por tarea t:

- **ER-ACE**: clases nuevas = las de la tarea t; memoria = buffer de clases antiguas + otras tantas imágenes de la tarea t, equilibradas (como ahora); máscara por origen.
- **WA**: antiguas = clases de T1…T(t−1), nuevas = Tt. Se aplica solo al modelo final.
- **cRT**: capa final reentrenada (2 épocas) con la memoria + otras tantas imágenes de T5. Solo modelo final.
- **Fuera**: LwF, ablación de fracciones, Dirichlet α = 0,1, referencia ×2.

### 1.5 Hiperparámetros fijos (no salen en la línea de órdenes)

Adam, lr 1e-3, lote 32, 5 épocas locales, 5 rondas por tarea, 10 clientes, ResNet-18 (conv 3 × 3, sin max-pooling), normalización + recorte con relleno 4 + volteo, cRT 2 épocas, calendario 0,75-0,5-0,25-0,25-0,25. Se definen en un único diccionario `FIJOS` de `cfl_tareas.py` y se copian a `config.json`.

La línea de órdenes solo expone variables de diseño: `--metodo`, `--buffers`, `--seed`, `--partition`, `--dataset`, `--joint`, `--quick`, `--audit`, `--only-phase1`, `--phase1-checkpoint`, `--output-dir`.

---

## 2. Métricas

| Archivo | Contenido |
|---|---|
| `matriz.csv` | Formato largo: `buffer, tras_tarea, tarea, acc`. R[t][i] = precisión en la tarea i (argmax sobre las 100 salidas) tras aprender la t |
| `history.csv` | Por ronda: `buffer, tarea, ronda, acc_vistas` y precisión de cada tarea vista |
| `results.csv` | Una fila por buffer con las métricas finales |
| `joint_results.csv` | Referencia conjunta |

Columnas de `results.csv`:

| Columna | Definición |
|---|---|
| `acc_final` | Precisión sobre el test de las 100 clases tras T5 (métrica principal) |
| `acc_incremental_media` | Media, para t = 1…5, de la precisión sobre las clases vistas tras la tarea t |
| `bwt` | Media para i = 1…4 de R[5][i] − R[i][i] |
| `olvido_medio` | Media para i = 1…4 de max_{t<5} R[t][i] − R[5][i] |
| `acc_t1_final`, `acc_t5_final` | Tarea más antigua y la última |
| `pred_nuevas_en_antiguas` | % de predicciones de T5 sobre el test de T1-T4 (sesgo hacia lo reciente) |
| `wa_*`, `crt_*` | `acc_final`, `acc_t1_final`, `acc_t5_final` tras WA y cRT |
| `memoria_por_clase_final` | Imágenes por clase y cliente al final (media) |

`fl_core.evaluate` se generaliza con `tasks=[[clases], …]` para devolver la precisión de cada tarea en una sola pasada (sin romper la firma actual con `group_a_classes`).

---

## 3. Dataset como parámetro

Nuevo `datasets_cfl.py` con una especificación por dataset: id y revisión en Hugging Face, columnas de imagen y etiqueta, número de clases, media y desviación, tamaño de imagen, tareas. Ahora solo `cifar100`; Tiny-ImageNet se añade en la fase 7 sin tocar el resto. `Preprocess` ya acepta `mean`/`std`.

---

## 4. Fases de implementación

| Fase | Trabajo | Resultado comprobable | Estimación |
|---|---|---|---|
| 1 | `datasets_cfl.py` + carga por tareas + `FIJOS` + CLI | `--quick` carga 5 tareas y escribe `data_manifest.json` con 5 × 20 clases | 1-2 h |
| 2 | Memoria por clase (construcción, recorte, anidamiento) + `evaluate` por tareas | Tests: tamaño fijo, cuota por clase, 10 % ⊂ 20 % tras cada tarea | 2 h |
| 3 | Bucle de tareas con los 4 métodos, WA, cRT, referencia conjunta, `matriz.csv` y métricas | `--quick` termina con los 4 métodos y la referencia | 3-4 h |
| 4 | Manifiesto de la fase 1 (contrato con `tareas` y `dataset`) | Reutilizar T1 da los mismos resultados que entrenarlo (test) | 1 h |
| 5 | Orquestador: clave `script` en el plan; `planes/plan_tareas.json` y `planes/plan_prueba_tareas.json`; informe nuevo `informe_tareas.py` (tabla por método y buffer, curva de precisión incremental, matriz R como mapa de calor) | `.\run.ps1 -dev plan planes/plan_prueba_tareas.json` termina con `INFORME.md` | 2-3 h |
| 6 | `docs/variables.md` (punto 5 del profesor) + README | Tabla de cada opción y columna con su línea de código | 1 h |
| 7 | *(si todo va bien)* Tiny-ImageNet: especificación, precarga en Docker, prueba `--quick` | Plan de 5 × 40 o 10 × 20 listo para lanzar | 2-3 h + GPU |

Tests nuevos en `test_tareas.py`, con datos sintéticos y sin red, como `test_cifar100_pipeline.py`. Los 58 tests actuales deben seguir pasando.

---

## 5. Ejecución y validación

1. **Prueba en miniatura** (CPU, minutos): `plan_prueba_tareas.json` con `--quick`.
2. **Piloto en GPU**: semilla 42, `equilibrados`, para medir el tiempo real y comprobar valores razonables:
   - sin replay, `acc_final` ≈ 20 % (solo T5 sobrevive) y `acc_t1_final` ≈ 0 %;
   - con replay, `acc_t1_final` > 0 y `acc_final` por debajo de la referencia conjunta.
3. **Plan completo**: 5 trabajos × 3 semillas + 6 fases 1 (IID y Dirichlet) = 21 tareas. Estimación sin medir: **~1 día de GPU** en la RTX 5050 con `vigilar_gpu.ps1 -Paralelo 2`.

## 6. Riesgos

- **Memoria muy pequeña al final** (2-3 imágenes por clase y cliente con 20 %): es lo esperado con memoria fija; si el replay deja de servir, se compara con la memoria creciente como ablación, no como cambio del protocolo.
- **f = 4/5 en T5**: las clases nuevas reciben solo 6 de 32 muestras por lote; puede bajar mucho la precisión de T5. Lo dirá el piloto.
- **Dirichlet con clases ausentes**: un cliente sin datos de la tarea t no entrena ni se promedia en esa tarea (como ahora).
