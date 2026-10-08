# Estado experimental

*Última revisión: 8 de octubre de 2026.* Este documento resume en un solo sitio las conclusiones de los informes de `resultados/`. Los informes originales se conservan sin cambios como registro detallado.

## Conclusión actual

**Todavía no hay ningún resultado válido sobre el efecto del tamaño del replay buffer.** Todas las ejecuciones verificadas terminan con precisión A = 0 % tras la fase B, con y sin replay, así que la «pérdida de precisión» es igual en todas las variantes y no permite compararlas. Las métricas están bien calculadas: se han recalculado desde los pesos con un evaluador independiente. El problema está en el régimen de entrenamiento, no en el cálculo.

## Qué se ha comprobado

| Ejecución | Base A | A después (0 % / con replay) | Qué demuestra | Informe |
| --- | ---: | ---: | --- | --- |
| Quick (2 clientes, 1 ronda) | 2 % | 0 % / 0 % | Solo que el flujo funciona; el modelo no aprende | [quick](resultados/cifar100_quick_2026-10-02_071246/INFORME.md), [auditoría](resultados/cifar100_auditoria_2026-10-02/INFORME.md) |
| Piloto validation (2 clientes, 3×3) | 27,2 % | 0 % / 0 % | Buffer mal dimensionado (200 imágenes = 0,8 % de A) y el modelo falla en sus propias imágenes de replay | [validación](resultados/cifar100_validacion_2026-10-02/INFORME.md), [denominador](resultados/cifar100_pool_corregido_2026-10-02/INFORME.md) |
| Completo 0 % / 20 %, 1 época local | 26,88 % | 0 % / 0 % | El buffer (4.995 imágenes) sí se usa, pero el modelo predice siempre clases B | [completo](resultados/cifar100_completo_0_20/INFORME.md) |
| Registros antiguos, 5 épocas locales (sin semilla fija) | ≈ 52 % | 4,5 % (5 %) / 19,4 % (15 %) | Con una base más fuerte el replay sí retiene algo | `resultados/historicos/results_step_six_macbook.txt` |

## Causas identificadas

1. **Base A débil.** Con 1 época local la fase A llega al 26,88 % y en ese régimen todas las variantes caen a 0. Con 5 épocas (A ≈ 52 %) los registros antiguos sí muestran retención.
2. **Sesgo hacia las clases nuevas.** Con una única cabeza de 100 clases y lotes que mezclan A y B al azar, un buffer del 20 % solo aporta ≈ 16,6 % de cada lote. El modelo final predice B en 5000/5000 imágenes de test A. Pero la precisión de A *restringida a su grupo* pasa de 1,86 % a 17,12 % con el buffer, y un control local con lotes 50/50 sube A de 0 % a 9,1 %.
3. **La métrica toca fondo.** `pérdida = A antes − A después` es idéntica en todas las variantes cuando todas acaban en 0.
4. **Sin normalización ni aumento de datos** en CIFAR-100, lo que limita la precisión base.

## Cambios introducidos (8 de octubre de 2026)

- `fl_core.py` centraliza modelo, FedAvg, entrenamiento, evaluación y preprocesado (antes estaban copiados en seis scripts).
- Normalización con las medias de CIFAR-100 y aumento de datos (recorte con relleno y volteo), activados por defecto. `--no-normalize --no-augment` reproduce el protocolo antiguo y es obligatorio para reutilizar `global_model_phase1.pt`.
- Replay controlado (mismos pesos, buffers anidados, mismas actualizaciones) por defecto en `cifar100_three_buffer.py`, y `--equal-steps` por defecto en DomainNet y MVTec.
- Nuevas variantes: `--replay-mix balanced` (proporción fija del buffer en cada lote) y `--distill-weight` (destilación sobre las clases A, estilo LwF).
- Nuevas métricas en `results.csv`: precisión en las 100 clases, retención (A después / A antes), precisión de A restringida a su grupo, proporción de predicciones B sobre el test A y entropía cruzada de A.
- Cada ejecución que entrena la fase A guarda `phase1_checkpoint.pt` junto a un manifiesto, para poder reutilizarla con trazabilidad.
- Imagen Docker reproducible con dependencias fijadas, CIFAR-100 incluido (funciona sin red) y el código dentro; cada `config.json` registra la revisión del código y una huella de los paquetes. Interfaz única: `.\run.ps1 info | tests | rapido | comparar | semillas`.

Las ejecuciones anteriores al 8 de octubre se hicieron **sin** normalización ni aumento: sus números no son directamente comparables con las nuevas.

## Próximos experimentos, en orden

1. **Base fiable.** `.\run.ps1 comparar` (0 % frente a 20 %, 5 épocas locales). Antes de comparar buffers, comprobar que A supera ≈ 60–65 %.
2. **Barrido con varias semillas.** `.\run.ps1 comparar --seeds 42 43 44 --buffers 0 5 10 20`.
3. **Lotes equilibrados.** Lo mismo con `--replay-mix balanced`.
4. **Destilación.** `--replay-mix balanced --distill-weight 1`, y su combinación con los buffers.
5. Interpretar con la retención y la precisión por grupo además de la pérdida, y con media ± desviación típica (`agregar_semillas.py`).

Ninguno de estos experimentos se ha ejecutado todavía con el código nuevo. Los resultados no están garantizados: no se ha introducido ninguna condición que fuerce una curva concreta.
