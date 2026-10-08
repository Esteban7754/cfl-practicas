# Resultados históricos

Gráficos y registros de ejecuciones anteriores a fijar la semilla y el entorno (antes del 2 de octubre de 2026). Se conservan sin modificar. **No son mediciones válidas para comparar buffers.**

| Archivo | Procedencia y advertencias |
| --- | --- |
| `results_step_six_macbook.txt` | Salida de `step_six_v1.py` en un MacBook (MPS), sin semilla fija (antes se llamaba `results`). La tabla resumen siempre decía «5 %» y «15 %»: el primer bloque corresponde en realidad al 25 % y al 50 % (ver las cabeceras «RUNNING STEP 6 EXPERIMENT»). |
| `step_six_v1_results.txt` | Otra salida de `step_six_v1.py`, con el mismo problema de etiquetas. |
| `group_a_accuracy_loss_all_buffers.png` | La barra «No Replay» **no se midió**: la versión antigua de `three_buffer_v2.py` suponía A = 0 % tras la fase B. |
| `group_a_accuracy_loss_5_15.png`, `group_a_accuracy_loss_all_buffers_v3.png` | CIFAR-100, una sola ejecución sin semilla fija. `three_buffer_size.py` fijaba además A = 25,06 % a mano y limitaba el eje a 30. |
| `domainnet_accuracy_loss_*.png`, `mvtec_continual_loss_*.png` | DomainNet y MVTec, una sola ejecución sin semilla fija. |
| `run_logs/step_four.log` | Registro del entrenamiento de `global_model_phase1.pt` (A = 26,88 %). Es la procedencia citada en `global_model_phase1.config.json`. |

Las ejecuciones actuales guardan sus resultados en carpetas propias dentro de `resultados/`.
