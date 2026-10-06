"""Genera el informe exclusivamente a partir de los artefactos ejecutados y verificados."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
verification = json.loads((ROOT / "verification.json").read_text())
audit = json.loads((ROOT / "audit.json").read_text())
config = json.loads((ROOT / "config.json").read_text())
diagnostic = json.loads((ROOT / "diagnostico_muestreo/diagnostico.json").read_text())
assert all(verification["checks"].values())
assert set(diagnostic["variants"]) == {"mezcla_actual", "lotes_equilibrados"}
assert diagnostic["source_checkpoint_sha256"] == hashlib.sha256((ROOT / "audit_phase2_20.pt").read_bytes()).hexdigest()
with (ROOT / "results.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
lines = [
    "# Comprobación ejecutada del replay 0 % / 20 % — 2 de octubre de 2026", "",
    "La ejecución completa y la reevaluación independiente confirman que ambos modelos terminan con accuracy A = 0 %. El buffer sí participa en el entrenamiento. La igualdad del «accuracy loss» se debe a que ambos modelos eligen siempre clases B y la accuracy A queda en su suelo. Esto no demuestra que la memoria no conserve información de A.", "",
    "## Resultados federados reales", "",
    "Se reutilizó el checkpoint A con 26,88 % de accuracy sobre las 5.000 imágenes de test A. Se usaron los 50.000 ejemplos de entrenamiento, 10 clientes, 5 rondas, 1 época local, batch 32, Adam y lr = 0,001, semilla 42. Ambas variantes empiezan con los mismos pesos y realizan las mismas actualizaciones por cliente/ronda. Con replay se limita el número de actualizaciones, por lo que no se recorren todas las imágenes del conjunto combinado en cada ronda.", "",
    "| Memoria | Accuracy A antes | Accuracy A después | Accuracy B después | Caída A |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(f"| {row['fraction']} | {float(row['step3_acc_a']):.2f} % | {float(row['step4_acc_a']):.2f} % | {float(row['step4_acc_b']):.2f} % | {float(row['loss_a']):.2f} pp |")
lines += ["", "La caída A es `accuracy A antes − accuracy A después`, en puntos porcentuales. No es la entropía cruzada ni una pérdida expresada como porcentaje.", "",
          "## Comprobación de memoria y entrenamiento", "",
          "| Memoria | Imágenes A conservadas / conjunto A | Actualizaciones | Presentaciones A | Presentaciones B |",
          "| --- | ---: | ---: | ---: | ---: |"]
for percent in (0, 20):
    memory = audit["replay_memory"][str(percent)]
    exposure = verification["training_exposure"][str(percent)]
    lines.append(f"| {percent} % | {memory['buffer_samples']:,} / {memory['a_pool_samples']:,} | {exposure['optimizer_steps']:,} | {exposure['a_samples_processed']:,} | {exposure['b_samples_processed']:,} |")
lines += ["", "El buffer del 20 % contiene 4.995 imágenes: 19,98 % del conjunto A completo, tras redondear por cliente. Se usaron muestras A en cada cliente y ronda, y se cubrieron sus 50 clases. Las presentaciones cuentan repeticiones durante el aprendizaje, no imágenes únicas.", "",
          "El buffer anterior del piloto tenía 200 imágenes respecto a las 25.000 aprendidas originalmente: era un 0,8 % del conjunto original. Corregir ese error no ha bastado para evitar el cero en este protocolo completo.", "",
          "## Qué oculta la accuracy global", "",
          "| Medida sobre test A | Sin memoria | Con 20 % |",
          "| --- | ---: | ---: |"]
no_memory = verification["evaluations"]["phase2_0"]["A"]
with_memory = verification["evaluations"]["phase2_20"]["A"]
lines += [
    f"| Predicciones que pertenecen a B | {no_memory['predictions_group_b']} / {no_memory['total']} | {with_memory['predictions_group_b']} / {with_memory['total']} |",
    f"| Entropía cruzada A | {no_memory['mean_cross_entropy']:.4f} | {with_memory['mean_cross_entropy']:.4f} |",
    f"| Accuracy A dando el grupo correcto (diagnóstico) | {no_memory['diagnostic_accuracy_with_true_group_given_percent']:.2f} % | {with_memory['diagnostic_accuracy_with_true_group_given_percent']:.2f} % |", "",
    "El diagnóstico restringe las predicciones a las 50 clases A utilizando el grupo verdadero. No es la accuracy del problema de 100 clases y no sustituye los resultados del CSV. La diferencia muestra reconocimiento de clases A conservado por la memoria, aunque la selección global siga favoreciendo B.", "",
    "## Control local del muestreo", "",
    "Se partió del mismo checkpoint final del 20 % en ambos controles. Se entrenó únicamente el cliente 0, con sus mismas 498 muestras de memoria y 2.506 muestras B, durante 79 actualizaciones con el mismo Adam/lr. La mezcla actual baraja el conjunto combinado; el control equilibrado toma 16 muestras A y 16 B por lote, con reposición. No hay agregación federada en este diagnóstico y el test no se usa para entrenar ni seleccionar parámetros.", "",
    "| Control local | Presentaciones A | Presentaciones B | Accuracy A | Accuracy B |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for mode, label in (("mezcla_actual", "Mezcla actual"), ("lotes_equilibrados", "Lotes A/B al 50/50")):
    metrics = diagnostic["variants"][mode]
    assert metrics["optimizer_steps"] == 79 and metrics["fc_weight_delta_l2"] > 0
    lines.append(f"| {label} | {metrics['a_samples_processed']} | {metrics['b_samples_processed']} | {metrics['test']['A']['accuracy_percent']:.2f} % | {metrics['test']['B']['accuracy_percent']:.2f} % |")
balanced_metrics = diagnostic["variants"]["lotes_equilibrados"]["test"]
natural_metrics = diagnostic["variants"]["mezcla_actual"]["test"]
lines += ["", f"En este control, los lotes equilibrados alcanzan A = {balanced_metrics['A']['accuracy_percent']:.2f} % frente a {natural_metrics['A']['accuracy_percent']:.2f} % con la mezcla actual, y B = {balanced_metrics['B']['accuracy_percent']:.2f} % frente a {natural_metrics['B']['accuracy_percent']:.2f} %. Hay recuperación de A y coste en B. La intervención cambia la proporción y el muestreo con reposición; no aísla sus efectos individuales.", "",
          "Estos valores son controles locales adicionales; no se mezclan con la tabla de resultados federados. El tamaño del buffer y la proporción de muestras antiguas en cada lote son parámetros diferentes: combinar 4.995 A con 25.000 B da aproximadamente un 16,65 % de muestras A, no una mezcla equilibrada.", "",
          "## Verificación y límites", "",
          "El verificador reconstruyó los datos y reevaluó los tres checkpoints con un cálculo independiente de accuracy. Todas las columnas del CSV coinciden. Comprobó pesos iniciales comunes, actualizaciones reales, exposición efectiva al replay y correspondencia de las particiones con las imágenes, etiquetas y multiplicidades del split oficial.", "",
          "Se encontraron 10 imágenes exactas compartidas entre los splits oficiales: 6 en test A y 4 en test B. Se conservan los splits originales para la ejecución principal; también se recalcularon las métricas excluyendo esas diez imágenes:", "",
          "| Modelo | Accuracy A sin duplicados | Accuracy B sin duplicados |",
          "| --- | ---: | ---: |"]
for phase, label in (("phase1", "Inicial A"), ("phase2_0", "Sin memoria"), ("phase2_20", "20 % memoria")):
    metrics = verification["evaluations"][phase]
    lines.append(f"| {label} | {metrics['A']['without_exact_train_duplicates']['accuracy_percent']:.4f} % | {metrics['B']['without_exact_train_duplicates']['accuracy_percent']:.4f} % |")
lines += ["", "Los duplicados no explican el cero. El protocolo del checkpoint original se reconstruyó desde `step_four.py` y su registro; no se dispone de un manifiesto generado durante aquel entrenamiento. Su SHA-256 y su accuracy original coinciden. Se ha ejecutado una semilla, por lo que no se establece una conclusión general sobre todos los tamaños de memoria.", "",
          "Para una siguiente comparación de retención, separar tamaño de memoria y proporción de replay por lote; comprobar primero lotes equilibrados en el entrenamiento federado completo, manteniendo el presupuesto común, y repetir con varias semillas. El control local prueba esa intervención en un cliente; no garantiza el resultado federado. Las métricas deben incluir accuracy A/B, entropía cruzada y reparto de predicciones para detectar el suelo de accuracy.", "",
          "## Artefactos", "",
          "- `results.csv`, `history.csv`: resultados finales y evolución por ronda.",
          "- `config.json`, `data_manifest.json`, `audit.json`: configuración, memoria y entrenamiento real.",
          "- `audit_phase1.pt`, `audit_phase2_0.pt`, `audit_phase2_20.pt`: checkpoints reevaluados.",
          "- `verification.json`, `verificacion.log`: verificación independiente y diagnóstico de predicciones.",
          "- `comprobar_solapamiento.py`, `solapamiento_oficial.json`: comprobación de duplicados originales.",
          "- `diagnosticar_muestreo.py`, `diagnostico_muestreo/diagnostico.json`, `diagnostico_muestreo.log`: control local y sus pesos.",
          "- `ejecucion.log`: registro completo del experimento.",
          "- `SHA256.json`: hashes de código y artefactos al generar este informe.", ""]
(ROOT / "INFORME.md").write_text("\n".join(lines), encoding="utf-8")
files = [path for path in ROOT.rglob("*") if path.is_file() and path.name != "SHA256.json"
         and "__pycache__" not in path.parts]
files += [ROOT.parents[1] / name for name in ("cifar100_three_buffer.py", "verificar_cifar100.py",
                                           "cifar100_sampling.py", "comparar_replay.ps1")]
hashes = {str(path.relative_to(ROOT.parents[1])).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in files}
(ROOT / "SHA256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
print("Informe generado desde resultados verificados y control local terminado.")
