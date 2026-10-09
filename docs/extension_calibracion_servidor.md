# Extensión elegida: calibración del clasificador en el servidor (opción B)

*Decidido el 9 de octubre de 2026. Se ejecuta después del hito 7 (análisis v2 con los resultados de Kaggle y de la RTX 5050).*

## Por qué esta extensión

Los resultados actuales muestran que, tras aprender la tarea B, el modelo **conserva conocimiento de A** pero la **última capa se inclina hacia las clases nuevas**:

- Sin replay, la precisión en A «sabiendo el grupo» es 12,9 % (47,2 % con destilación), aunque el 100 % de las predicciones sobre imágenes de A son clases B.
- Las correcciones de la última capa ayudan: Weight Aligning sube la mezcla normal hasta el nivel de los lotes equilibrados, y el plan de la GPU mide además cRT.

Si el problema está sobre todo en la última capa, el **servidor** podría corregirla sin ver datos de los clientes, solo con estadísticas agregadas. Encaja con la privacidad del aprendizaje federado y aprovecha que el servidor ve a todos los clientes.

## Idea de partida (a concretar)

1. Cada cliente envía, además de sus pesos, **estadísticas por clase de las representaciones** de la penúltima capa (media y, quizá, covarianza diagonal) de los datos que tiene en ese momento. De las clases antiguas, las estadísticas guardadas cuando las aprendió o las de su buffer.
2. El servidor **agrega** esas estadísticas, genera **representaciones sintéticas** por clase (por ejemplo, gaussianas) y **reentrena solo la última capa** con un conjunto equilibrado entre clases antiguas y nuevas.
3. Variantes a comparar: con y sin buffer local; estadísticas de primer orden frente a segundo orden; recalibrar en cada ronda o solo al final de cada tarea.

## Trabajo relacionado que hay que revisar antes (novedad)

- **CCVR** (Luo et al., NeurIPS 2021, «No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data»): calibra el clasificador en el servidor con representaciones virtuales a partir de estadísticas de los clientes, pero para **datos no IID sin tareas secuenciales**. La aportación aquí sería llevarlo al **aprendizaje continuo incremental por clases**, donde las clases antiguas ya no están en los datos actuales.
- Métodos federados incrementales por clases (GLFC, TARGET, FedCIL, LANDER…) y correcciones del clasificador en aprendizaje continuo centralizado (WA, BiC, cRT, ER-ACE).
- Comprobar qué hacen con la privacidad de las estadísticas (las medias por clase pueden filtrar información).

## Experimentos previstos

- Mismo protocolo que el estudio actual (CIFAR-100, 10 clientes, FedAvg, presupuesto controlado), con **2 tareas y con 5 tareas** (extensión A como base).
- Comparar con: sin corrección, WA, cRT local, lotes equilibrados y ER-ACE.
- Repartos IID y Dirichlet (α = 0,5 y 0,1).
- Métricas: precisión por tarea y en todas las clases, BWT, proporción de predicciones hacia clases nuevas, coste de comunicación de las estadísticas.

## Requisitos antes de empezar

- Hito 7 cerrado: resultados de 5 semillas y comparación T4 frente a RTX 5050.
- Mirar en esos resultados cuánto aporta cRT: si un cRT local ya cierra casi toda la brecha, la versión en el servidor tiene margen pequeño y conviene replantear.
- Implementar primero la extensión A (más de dos tareas), porque B se evaluará sobre ella.
