# Variables del protocolo de 5 tareas

Qué es cada variable de `cfl_tareas.py`, cómo se calcula y dónde está en el código. Es la referencia para revisar el diseño (punto 5 de lo acordado con el profesor, 25/10/2026). El protocolo completo está en [plan_cinco_tareas.md](plan_cinco_tareas.md).

**Notación.** T = número de tareas (5). t = tarea, contando desde 1. R[t][i] = precisión en el test de la tarea i después de aprender la tarea t (argmax sobre las 100 salidas). Precisión en %.

---

## 1. Variables de diseño (línea de órdenes)

Son las únicas que cambian entre ejecuciones.

| Opción | Valores | Qué controla | Código |
|---|---|---|---|
| `--metodo` | `concat`, `balanced`, `calendario`, `ace` | Cómo se combina la memoria con la tarea actual en cada lote (§3) | [cfl_tareas.py:95](../cfl_tareas.py#L95), [batches:291](../cfl_tareas.py#L291) |
| `--buffers` | Lista de %, debe incluir 0 | Tamaño de la memoria de cada cliente; se ejecuta un experimento por valor | [cfl_tareas.py:96](../cfl_tareas.py#L96) |
| `--seed` | Entero (42, 43, 44) | Reparto de datos, inicialización, orden de lotes, aumento y memoria | [cfl_tareas.py:98](../cfl_tareas.py#L98) |
| `--partition` | `iid`, `dirichlet` | IID: cada cliente recibe ~1/10 de cada clase. Dirichlet: reparto desigual por clase con α = 0,5 | [cfl_tareas.py:99](../cfl_tareas.py#L99) |
| `--dataset` | `cifar100` | Dataset y división en tareas ([datasets_cfl.py](../datasets_cfl.py)) | [cfl_tareas.py:100](../cfl_tareas.py#L100) |
| `--joint` | — | Añade la referencia conjunta (§5) | [cfl_tareas.py:101](../cfl_tareas.py#L101) |
| `--quick` | — | Prueba funcional: 2 clientes, 1 época, 1 ronda por tarea, 100 imágenes por tarea y cliente | [QUICK:58](../cfl_tareas.py#L58) |
| `--only-phase1` / `--phase1-checkpoint` | — / ruta | Entrenar la tarea 1 una vez y reutilizarla (validada con su manifiesto) | [cfl_tareas.py:105](../cfl_tareas.py#L105) |
| `--audit` | — | Guarda en `audit.json` los pasos y etiquetas de cada cliente y ronda, y el tamaño de cada memoria | [cfl_tareas.py:104](../cfl_tareas.py#L104) |

## 2. Hiperparámetros fijos (`FIJOS`)

No se pueden cambiar por línea de órdenes; se copian a `config.json` en la clave `fijos`. Definidos en [cfl_tareas.py:42](../cfl_tareas.py#L42).

| Variable | Valor | Significado |
|---|---|---|
| `num_clients` | 10 | Clientes federados; todos participan en cada ronda |
| `batch_size` | 32 | Imágenes por lote |
| `local_epochs` | 5 | Épocas locales por ronda (fija el número de pasos, §3) |
| `rounds_per_task` | 5 | Rondas federadas por tarea → 25 en total |
| `lr`, `optimizer` | 0,001, Adam | El optimizador se crea de nuevo en cada ronda y cliente |
| `model` | ResNet-18 | Primera convolución 3 × 3 y sin max-pooling, 100 salidas ([fl_core.get_model](../fl_core.py#L40)) |
| `normalize`, `augment` | Sí, sí | Media y desviación de CIFAR-100; recorte con relleno de 4 y volteo horizontal |
| `crt_epochs` | 2 | Épocas de cRT (§4) |
| `calendario` | 0,75 · 0,5 · 0,25 · 0,25 · 0,25 | Fracción de memoria por lote en cada ronda, con el método `calendario` (igual en cada tarea) |
| `dirichlet_alpha` | 0,5 | Concentración de Dirichlet (menor = clientes con menos clases) |

## 3. Variables derivadas (internas)

| Variable | Fórmula | Ejemplo (IID, 20 %) | Código |
|---|---|---|---|
| Tareas | Tarea t = clases 20·(t−1) … 20·t − 1 | T1 = 0-19, T5 = 80-99 | [datasets_cfl.py:31](../datasets_cfl.py#L31) |
| Capacidad de la memoria | `buffer % · imágenes del cliente / T` | 20 % · 5 000 / 5 = **200** por cliente (2 000 en total) | [memoria_cfl.py:70](../memoria_cfl.py#L70) |
| Cuota por clase | Capacidad repartida por turnos entre las clases vistas, en un orden de clases fijo del cliente | 10 (tras T1), 5, 3-4, 2-3, 2 | [memoria_cfl.py:20](../memoria_cfl.py#L20) |
| Pasos por ronda | `local_epochs · ceil(n_tarea_cliente / 32)`, igual con cualquier buffer y método | 5 · ceil(1 000 / 32) = 160 | [cfl_tareas.py:255](../cfl_tareas.py#L255) |
| Fracción de memoria `f` | `balanced` y `ace`: (t−1)/t; `calendario`: `FIJOS["calendario"][ronda]` | T2: 1/2 → 16 de 32; T5: 4/5 → 26 de 32 | [cfl_tareas.py:302](../cfl_tareas.py#L302) |
| Memoria de ER-ACE | Memoria de clases antiguas + otras tantas imágenes de la tarea actual (equilibradas por clase) | 200 + 200 | [cfl_tareas.py:307](../cfl_tareas.py#L307) |

**Cómo se forman los lotes.**
- `concat`: tarea actual y memoria se barajan juntas y se recorren por épocas, cortando al llegar a los pasos de la ronda. La memoria ocupa en el lote su proporción natural: 200 / 1 200 ≈ 17 %.
- `balanced` y `calendario`: cada lote lleva `round(32·f)` imágenes de la memoria (con reposición) y el resto de la tarea actual.
- `ace`: igual que `balanced`, pero la pérdida enmascara las clases antiguas en las imágenes entrantes; las de la memoria usan la entropía cruzada completa.

**Cómo evoluciona la memoria.** Al terminar cada tarea se recalculan las cuotas, se recorta cada clase antigua (se queda con un prefijo de sus imágenes, en un orden fijo) y se añaden las clases nuevas. Lo descartado no vuelve, y con la misma semilla la memoria del 10 % está siempre contenida en la del 20 %.

## 4. Salidas

### `results.csv` (una fila por buffer)

| Columna | Definición | Código |
|---|---|---|
| `acc_final` | Precisión sobre el test de las 100 clases tras la última tarea. **Métrica principal** | [cfl_tareas.py:374](../cfl_tareas.py#L374) |
| `acc_incremental_media` | Media, para t = 1…T, de la precisión sobre las clases vistas tras la tarea t | [:375](../cfl_tareas.py#L375) |
| `bwt` | Media para i < T de R[T][i] − R[i][i]: cuánto cambia cada tarea desde que se aprendió (negativo = olvido) | [:376](../cfl_tareas.py#L376) |
| `olvido_medio` | Media para i < T de max_{t<T} R[t][i] − R[T][i]: caída desde el mejor valor de cada tarea | [:377](../cfl_tareas.py#L377) |
| `acc_t1_inicial`, `acc_t1_final` | Tarea 1 justo después de aprenderla y al final | [:378](../cfl_tareas.py#L378) |
| `acc_t5_final` | Última tarea al final (mide lo que se aprende de lo nuevo) | [:378](../cfl_tareas.py#L378) |
| `pred_nuevas_en_antiguas` | % de imágenes de test de T1-T4 que el modelo final asigna a una clase de T5 (sesgo hacia lo reciente) | [:380](../cfl_tareas.py#L380) |
| `memoria_total`, `memoria_por_clase_final` | Imágenes en memoria (suma de clientes) y media por clase y cliente durante la última tarea | [:381](../cfl_tareas.py#L381) |
| `wa_gamma`, `wa_acc_final`, `wa_acc_t1_final`, `wa_acc_t5_final` | El mismo modelo tras Weight Aligning: las filas de la capa final de T5 se escalan por γ = norma media antiguas / norma media nuevas | [post_hoc:385](../cfl_tareas.py#L385) |
| `crt_acc_*` | Tras cRT: capa final reentrenada 2 épocas (resto congelado) con la memoria + otras tantas imágenes de T5. Solo con buffer > 0 | [post_hoc:385](../cfl_tareas.py#L385) |
| `gap_vs_joint` | Referencia conjunta − `acc_final` (solo con `--joint`) | [:457](../cfl_tareas.py#L457) |

### Otros archivos

| Archivo | Contenido |
|---|---|
| `matriz.csv` | `buffer, tras_tarea, tarea, acc`: la matriz R completa ([record_task:319](../cfl_tareas.py#L319)) |
| `history.csv` | `buffer, tarea, ronda, acc_vistas, acc_t1…`: evolución ronda a ronda ([:349](../cfl_tareas.py#L349)) |
| `joint_results.csv` | Referencia conjunta: `rounds`, `acc_final` y la precisión por tarea |
| `data_manifest.json` | Imágenes y clases presentes por cliente y tarea, preprocesado y reparto |
| `config.json` | Configuración completa (`fijos` incluidos), entorno y checkpoint usado |
| `phase1_checkpoint.config.json` | Manifiesto de la tarea 1: las claves de `CONTRATO` ([:64](../cfl_tareas.py#L64)), SHA-256 y precisión |

## 5. Referencia conjunta

Las 100 clases a la vez desde cero, sin tareas: T · `rounds_per_task` = 25 rondas, y en cada ronda los pasos de una tarea media del cliente (`local_epochs · ceil(n_cliente / T / 32)`). Así gasta el mismo presupuesto total que la secuencia de tareas. Es una cota orientativa, no un máximo estricto ([joint:422](../cfl_tareas.py#L422)).
