# Auditoría de los resultados CIFAR-100 — 2 de octubre de 2026

**Revisión posterior del diseño:** reproducir los porcentajes no validaba el experimento. El [piloto controlado y la evaluación sobre las propias imágenes de replay](../cifar100_validacion_2026-10-02/INFORME.md) confirman que faltaba comprobar aprendizaje suficiente. No hay todavía resultados aptos para concluir sobre el efecto del buffer.

Los números del informe anterior proceden de la ejecución real. Se han reproducido y recalculado desde los pesos con un evaluador independiente. No se ha encontrado una discrepancia de cálculo en las métricas. La prueba reducida, sin embargo, NO valida aprendizaje suficiente ni permite comparar replay buffers o medir olvido de forma útil.

La validación anterior comprobó que el flujo terminaba, pero faltaba comprobar que el modelo aprendía antes de interpretar su pérdida de precisión.

## Evidencia de reproducción

Se conservan los resultados originales. Dos nuevas ejecuciones, una sin instrumentación y otra con diagnósticos, produjeron exactamente el mismo CSV. Los tres archivos tienen el mismo SHA-256:

```text
70ab763dea5c00b54f44744afc9349e624a14bb56611d7978899968e30450f73
```

- Original: `../cifar100_quick_2026-10-02_071246/results.csv`.
- Repetición: `reproduccion/results.csv` y `reproduccion/ejecucion.log`.
- Instrumentada: `instrumentada/results.csv`, `instrumentada/ejecucion.log`, `instrumentada/audit.json` y los siete checkpoints `instrumentada/audit_phase*.pt`.

## Qué explica las métricas bajas

Cada cliente solo realiza 4 actualizaciones de Adam por fase. Sus 100 imágenes de A cubren 40 clases en un cliente y 45 en el otro, de las 50 posibles. Los buffers contienen únicamente 5, 10, 15, 20 o 25 imágenes por cliente. El test utiliza solo 100 imágenes por grupo, con 43 clases presentes en cada grupo.

Tras la primera fase, el modelo predice la clase 23 en 69 imágenes de A y la 27 en las otras 31: acierta 2 de 100. No ha aprendido una clasificación útil de las 50 clases.

Tras la segunda fase con el buffer del 10 %, predice la clase 67 en 99 imágenes de B y la 76 en la restante. La clase 67 no está presente en las etiquetas del test reducido de B, y la otra predicción también es incorrecta. De ahí el 0 % real, no un fallo de formato o un valor sustituido manualmente.

Todos los buffers tienen la misma precisión inicial de A, 2 %, porque parten del mismo checkpoint. Todos obtienen 0 aciertos en A después, por lo que `2 - 0 = 2` puntos porcentuales en todas las filas. Esa igualdad no demuestra equivalencia entre buffers.

## Comprobaciones adicionales superadas

El script `verificar_cifar100.py` carga las imágenes y etiquetas, construye los tensores directamente y evalúa los pesos sin utilizar `transform_batch`, `collate_fn` ni `evaluate` del experimento. Los recuentos de aciertos y los porcentajes coinciden con todas las filas del CSV.

También se comprobó que las seis variantes parten exactamente de los mismos pesos, que los 14 entrenamientos locales hacen 4 actualizaciones y cambian los pesos de la capa final, y que no hay imágenes idénticas entre los subconjuntos seleccionados de entrenamiento y test ni entre los subconjuntos A de los dos clientes.

El primer intento del evaluador independiente omitió el barajado inicial de test que hace Flower y, por ello, seleccionó otras 100 imágenes. Falló la comparación, como debía. Se corrigió el verificador para reproducir ese orden y comprobar también los recuentos de etiquetas. Se conserva ese intento en `verificacion_muestreo_inicial.log`; el registro válido es `verificacion.log`. Las métricas originales no se modificaron.

## Control positivo de aprendizaje

Con la misma arquitectura y optimizador, se entrenó 50 veces sobre el mismo lote REAL de 32 imágenes:

| Medida | Antes | Después |
| --- | ---: | ---: |
| Precisión sobre ese lote | 2/32 = 6,25 % | 32/32 = 100 % |
| Entropía cruzada durante entrenamiento | 4,536060 en el primer paso | 0,000245 en el último |

La precisión sobre las 100 imágenes de test A, separadas del entrenamiento, fue 8 %. El 100 % es memorización del lote, no precisión de generalización ni resultado del experimento federado. Este control demuestra que el modelo y la actualización de pesos pueden aprender; no valida la comparación de buffers.

Datos completos: `instrumentada/verification.json`.

## Cómo hacer pruebas útiles

Para comprobar funcionamiento y poder auditar las métricas:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\ejecutar.bat cifar100_three_buffer.py --quick --audit --output-dir resultados/nueva_auditoria
.\ejecutar.bat verificar_cifar100.py resultados/nueva_auditoria
```

Para estudiar replay, usa un entrenamiento con suficiente aprendizaje en A y B y un test que cubra todas las clases. La opción completa mantiene los 10 clientes, 5 rondas y 10 épocas locales del experimento original:

```powershell
.\ejecutar.bat cifar100_three_buffer.py --audit --output-dir resultados/cifar100_completo
```

Esa ejecución completa no se ha realizado durante esta auditoría. Para conclusiones sobre los buffers, además hay que repetir con varias semillas y controlar el muestreo: el script actual elige buffers independientes y avanza el generador aleatorio entre variantes, por lo que no hay garantía de mejora monotónica al aumentar el porcentaje.
