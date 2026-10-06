# Corrección del denominador de memoria — 2 de octubre de 2026

Se encontró un error de diseño en el piloto anterior: se reutilizó un checkpoint del experimento con todos los datos A, pero se calculó el replay sobre un recorte distinto. Los resultados anteriores no son una comparación de conservar el 20 % del conjunto aprendido en la fase A.

## Cantidades comprobadas

| Cantidad | Imágenes |
| --- | ---: |
| Entrenamiento CIFAR-100 completo | 50.000 |
| Grupo A completo, usado por `step_four.py` | 25.000 |
| Grupo B completo | 25.000 |
| Grupo A recortado en el piloto | 1.000 |
| Buffer que el piloto etiquetó como 20 % | 200 |

El buffer del piloto era `200 / 1.000 = 20 %` del recorte, pero **`200 / 25.000 = 0,8 %`** respecto al conjunto A del entrenamiento original. Un 20 % del conjunto A completo requiere unas 5.000 imágenes, con pequeñas diferencias por redondear el tamaño de cada cliente.

Se reevaluó el checkpoint original sobre las 5.000 imágenes de test A: **1.344 aciertos, 26,88 %**, coincidiendo con `.run_logs/step_four.log`. El SHA-256 del checkpoint coincide con el utilizado por el piloto. Las cantidades y recuentos están en `checkpoint_inspection.json`; el código y registro de esta comprobación se conservan en `inspeccionar.py` e `inspeccion.log`.

El protocolo de entrenamiento del checkpoint se reconstruyó a partir de `step_four.py` y su registro existente y se identificó por SHA-256 en `global_model_phase1.config.json`. No es un manifiesto creado durante el entrenamiento original; esa limitación queda explícita en el archivo.

## Qué se corrigió

- Reutilizar un checkpoint exige una configuración compatible de clientes, semilla, revisión de datos, muestreo y recorte del conjunto A. La comparación anterior se rechaza antes de entrenar o crear resultados.
- Cada ejecución registra en `data_manifest.json` el tamaño del conjunto A. `audit.json` registra la cantidad real de memoria total y por cliente.
- `--controlled-replay` permite comparar todos los datos con los mismos pesos iniciales, semillas, buffers anidados y número de actualizaciones.
- `comparar_replay.ps1` deja preparada la comparación 0 % frente a 20 % con los datos completos: 10 clientes, 5 rondas y 1 época local, siguiendo el presupuesto de `step_four.py`. Se utiliza el mismo número de actualizaciones por cliente y ronda; con replay se dedica una parte de ese presupuesto a las muestras conservadas.
- El verificador admite esa comparación completa y comprueba tanto las métricas como la memoria respecto al conjunto A real.

Pasaron cinco tests, incluida la incompatibilidad entre checkpoint completo y recorte de replay. También se comprobó que el comando anterior se rechaza sin crear una carpeta de resultados.

## Interpretación

Este error invalida interpretar el piloto como una comparación de memoria del 0 % frente al 20 % de lo aprendido. **No demuestra por sí solo que el denominador sea la única causa de los ceros**, ni garantiza una curva distinta cuando se corrija. Hace falta ejecutar la comparación coherente y observar sus métricas; no se deben modificar los números ni exigir un resultado concreto para validar el código.

Para ejecutarla, con Docker abierto, desde la raíz del proyecto:

```powershell
.\comparar_replay.ps1
```

**Actualización:** la comparación completa 0 %/20 % ya se ejecutó y se verificó desde los checkpoints. Ambos modelos vuelven a obtener A = 0 %, aunque el replay sí utiliza muestras A y reduce su entropía cruzada. El [informe de la ejecución completa](../cifar100_completo_0_20/INFORME.md) contiene los resultados, el diagnóstico de predicciones y el control local con lotes equilibrados. Para repetir el comando, elige otra carpeta mediante `-OutputDir`; el lanzador conserva la ejecución existente.
