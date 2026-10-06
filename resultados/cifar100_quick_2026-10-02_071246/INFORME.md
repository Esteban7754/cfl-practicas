# Prueba reducida CIFAR-100 — 2 de octubre de 2026

**Revisado mediante auditoría:** las cifras se han reproducido y verificado desde los pesos. Esta prueba solo valida el funcionamiento del flujo, no aprendizaje suficiente ni comparaciones de replay. La precisión inicial de 2 % y los ceros posteriores corresponden a un modelo que apenas distingue clases. Consulta la [auditoría y el control de aprendizaje](../cifar100_auditoria_2026-10-02/INFORME.md).

La reconstrucción de la imagen `cfl-practicas`, la compilación de los scripts Python y `test_env.py` finalizaron correctamente. La comparativa reducida terminó con código de salida 0 y ejecutó los seis buffers. Se usó el dataset real cacheado, con la revisión indicada en `config.json`, en modo sin conexión.

Configuración: CPU, 4 hilos, semilla 42, 2 clientes, 1 ronda por fase, 1 época local, lote de 32, 100 muestras por grupo y cliente y 100 muestras de evaluación por grupo. Las precisiones son comprobaciones funcionales de una prueba pequeña; no sirven para concluir qué buffer funciona mejor ni sustituyen el experimento completo.

| Buffer | A antes (%) | A después (%) | B después (%) | Pérdida A (puntos porcentuales) |
| --- | ---: | ---: | ---: | ---: |
| 0 % | 2 | 0 | 3 | 2 |
| 5 % | 2 | 0 | 5 | 2 |
| 10 % | 2 | 0 | 0 | 2 |
| 15 % | 2 | 0 | 5 | 2 |
| 20 % | 2 | 0 | 0 | 2 |
| 25 % | 2 | 0 | 1 | 2 |

Archivos: `results.csv`, `config.json`, `ejecucion.log`, `group_a_accuracy_loss_up_to_20.png` y `group_a_accuracy_loss_all_buffers.png`.

Para repetir desde PowerShell, en la raíz del proyecto:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\ejecutar.bat test
.\ejecutar.bat cifar100_three_buffer.py --quick --output-dir resultados/otra_prueba
```

Para la comparativa completa, quita `--quick` y elige otra carpeta. Sus parámetros originales son 10 clientes, 5 rondas por fase y 10 épocas locales, con todos los datos; requiere mucho más tiempo.

Problemas encontrados y soluciones verificadas:

- Docker estaba apagado. Tras iniciar Docker Desktop, `docker build -t cfl-practicas .` funcionó y las pruebas pasaron.
- Python nativo no pudo cargar `torch.dll`: WinError 4551 por una directiva de Control de aplicaciones. La ejecución en Docker funcionó. No se modificaron las políticas de Windows.
- El menú de `run.bat` tenía saltos incondicionales tras los `if`; se corrigieron los bloques. El lanzador ahora monta la carpeta del proyecto y transmite los argumentos de los scripts. Se comprobó `ejecutar.bat cifar100_three_buffer.py --help` desde otra carpeta.
- MVTec no tiene el dataset local en `data/mvtec_anomaly_detection/`. Hay que añadir los datos antes de ejecutar ese experimento. DomainNet y MVTec no se entrenaron en esta prueba.
