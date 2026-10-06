# Revisión del diseño de las pruebas — 2 de octubre de 2026

**Corrección posterior:** se detectó que el 20 % se calculaba sobre un recorte de 1.000 imágenes, aunque el checkpoint procedía del experimento con las 25.000 imágenes A. Las 200 imágenes de memoria eran un 0,8 % respecto al conjunto original. El [informe de corrección del denominador](../cifar100_pool_corregido_2026-10-02/INFORME.md) documenta la comprobación y el comando actualizado. El comando de piloto con ese checkpoint completo que figura más abajo ahora se rechaza por incompatibilidad.

**Conclusión del piloto:** las cifras son reales, pero las pruebas reducidas no consiguieron un aprendizaje suficiente para evaluar la eficacia del replay. La ejecución exitosa y la reproducción del CSV no bastaban para validar el experimento.

**Comprobación posterior ejecutada:** el [informe de la comparación completa 0 %/20 %](../cifar100_completo_0_20/INFORME.md) contiene los nuevos resultados sobre todos los datos, su reevaluación independiente y el control del muestreo. El accuracy A vuelve a ser cero, pero ahora está comprobado que se usan 4.995 imágenes de memoria y se identifica que los modelos predicen siempre clases B.

## Por qué todos los buffers daban 2

La pérdida se calcula como precisión de A antes menos precisión de A después, en puntos porcentuales. La prueba quick tenía una precisión inicial de 2 % y una final de 0 % en todas las variantes. Por eso todas daban 2 puntos. Una igualdad por sí sola no demuestra un fallo de la fórmula: puede aparecer cuando todas las variantes olvidan completamente A. En este caso, además, la primera fase apenas había aprendido.

## Piloto controlado ejecutado

Para investigar se corrigieron el muestreo y las condiciones de comparación:

- 2 clientes, 500 imágenes por grupo y cliente, 10 de cada clase.
- Test fijo de 1.000 imágenes por grupo, 20 de cada una de las 50 clases.
- Checkpoint A existente, con SHA-256 registrado en `config.json`, reevaluado antes de usarlo.
- Replay 0 % frente a 20 %, buffers anidados, semilla reiniciada y mismo punto de partida.
- 3 rondas, 3 épocas locales y 48 actualizaciones por cliente y ronda en ambas variantes.

Resultados del diagnóstico, recalculados desde los pesos con otro evaluador:

| Replay | A antes (%) | A después (%) | B después (%) | Pérdida A (puntos) |
| --- | ---: | ---: | ---: | ---: |
| 0 % | 27,2 | 0 | 9,1 | 27,2 |
| 20 % | 27,2 | 0 | 8,7 | 27,2 |

Este piloto seguía sin discriminar retención: el modelo predecía clases de B para todas las imágenes de A. No demuestra que el replay no sirva ni que ambas capacidades sean equivalentes.

## Comprobación que faltaba: aprender las propias muestras

Se evaluaron también las imágenes efectivamente utilizadas para entrenar:

- Sin replay, el modelo acertaba solo 120 de las 1.000 imágenes de entrenamiento B: 12 %.
- Con replay del 20 %, acertaba **0 de las 200 imágenes del propio buffer**.

La prueba no puede presentarse como una comparación de capacidades de memoria cuando el modelo final falla incluso en las muestras conservadas para replay. Los controles iniciales (10 % mínimo en A y 8 % en el test de B) eran insuficientes para detectar ese problema.

Se investigó además BatchNorm: se recalibraron únicamente sus estadísticas usando las imágenes de entrenamiento disponibles para cada variante. No se utilizaron imágenes de test ni se cambiaron parámetros entrenables. B pasó de 9,1 % a 10,8 % sin replay y de 8,7 % a 9,6 % con replay; A y las imágenes del buffer siguieron en 0 %. Este diagnóstico no respalda que la normalización sea la causa principal del cero. Es una comprobación posterior, separada de los resultados originales, no un cambio oculto de la evaluación.

Registros: `ejecucion.log`, `history.csv`, `results.csv`, `audit.json`, `verification.json`, `verificacion_final.log`, `batchnorm.log` y `batchnorm_diagnostics.json`. Se conservan los gráficos como artefactos del diagnóstico; no son evidencia de equivalencia entre buffers. `validation.json` marca el piloto como inconcluyente tras esta revisión.

## Correcciones en las pruebas

- Quick queda marcado como prueba funcional y deja de generar gráficos comparativos.
- Validation comprueba cobertura de clases, misma base y presupuesto, además del aprendizaje en entrenamiento y en el buffer.
- Se declara un control operativo de al menos 50 % sobre entrenamiento B y sobre replay si existe. Ese umbral se añadió tras detectar el fallo del piloto; no se presenta como un requisito preregistrado de la ejecución anterior ni como certificación científica.
- Si falla un control, se conserva el diagnóstico y se evita publicar una comparativa como válida.
- El verificador separa las métricas numéricas correctas de un entrenamiento inconcluyente; en ese caso termina con código 2.
- Se comprobó también el rechazo del checkpoint quick: obtiene 2,5 % en el test equilibrado y se cancela antes de comparar buffers. Sus registros están en `rechazo_base_insuficiente/`.

Los cuatro tests de muestreo y controles pasaron. No se introduce ninguna condición que obligue a que la curva de loss disminuya o a que los buffers den resultados diferentes.

## Siguiente prueba con mayor presupuesto

La interfaz permite aumentar rondas y épocas manteniendo los controles y la igualdad del presupuesto:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\ejecutar.bat cifar100_three_buffer.py --validation --audit --phase1-checkpoint global_model_phase1.pt --buffers 0 20 --rounds 5 --local-epochs 10 --output-dir resultados/piloto_mayor
.\ejecutar.bat verificar_cifar100.py resultados/piloto_mayor
```

Esa configuración mayor todavía no se ha ejecutado; no se garantiza su resultado. Para evaluar el efecto del replay hay que conseguir aprendizaje suficiente con un presupuesto declarado, comprobar retención y aprendizaje nuevo por separado y repetir con varias semillas. El experimento completo original tampoco se ha ejecutado durante esta revisión.
