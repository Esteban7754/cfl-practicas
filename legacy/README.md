# Scripts históricos

Versiones anteriores de los experimentos de CIFAR-100, conservadas para poder reproducir los resultados y gráficos antiguos (`resultados/historicos/`). Para trabajo nuevo usa `cifar100_three_buffer.py` en la raíz.

| Script | Qué hacía |
| --- | --- |
| `step_four.py` | Fase A y fase B sin replay. Genera `global_model_phase1.pt` y `global_model_phase2.pt` en la carpeta desde la que se ejecuta. |
| `step_six_v1.py` | Buffers del 0 % al 20 %, repitiendo la fase A en cada variante. |
| `three_buffer_size.py` | Igual que el anterior, del 0 % al 25 %. |
| `three_buffer_v2.py` | Fase A una vez y buffers del 0 % al 25 % desde los mismos pesos. |

Se ejecutan desde la raíz del proyecto para que encuentren los módulos comunes. La imagen Docker define `PYTHONPATH=/workspace`:

```powershell
.\run.ps1 legacy/step_four.py
```

No usan normalización ni aumento de datos. Los que usan `experiment_common.py` (`three_buffer_v2.py`, `three_buffer_size.py`, `step_six_v1.py`) ahora tienen `--equal-steps` activado por defecto; `--no-equal-steps` recupera el comportamiento original, en el que más buffer también implicaba más actualizaciones.
