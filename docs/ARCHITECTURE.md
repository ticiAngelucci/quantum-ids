# Arquitectura

## Objetivo

`quantum-ids` compara un pipeline clásico de detección de anomalías contra un pipeline de `Quantum Machine Learning`, manteniendo además un flujo experimental `live` para trabajar con tráfico capturado en laboratorio.

La arquitectura actual separa cuatro responsabilidades:

1. entrenamiento clásico
2. entrenamiento cuántico
3. captura y preparación de tráfico live
4. visualización y operación desde dashboard

## Mapa del sistema

```text
data/dataset.csv
    ├── src/classical/train_model.py
    │     └── results/classical_metrics.json
    │     └── results/random_forest_model.joblib
    │     └── results/scaler.joblib
    │     └── results/pca.joblib
    │
    └── src/quantum/train_vqc_simulator.py
          └── results/quantum_simulated_metrics*.json

trafico live / pcap
    └── src/live_detection/capture.py
          └── src/live_detection/feature_extractor.py
                └── results/live_capture.csv
                └── results/live_training_dataset.csv
                      └── src/quantum/train_vqc_simulator.py --dataset-source live
                            └── results/quantum_live_simulated_metrics*.json

app/app.py
    └── app/dashboard/*
          ├── constants.py
          ├── theme.py
          ├── data.py
          ├── analytics.py
          ├── ui.py
          └── views.py
```

## Capas del proyecto

### 1. `src/classical`

Responsabilidad:

- preparar el flujo clásico
- detectar la columna objetivo
- convertir etiquetas a binario
- entrenar el baseline de `Random Forest`

Salida principal:

- artefactos serializados en `results/`

### 2. `src/preprocessing`

Responsabilidad:

- limpiar datasets
- escalar features
- aplicar `PCA`
- preparar el bundle de entrenamiento para QML

Pieza central:

- `quantum_preprocessing.py`

Entrega:

- `QuantumDatasetBundle`

### 3. `src/quantum`

Responsabilidad:

- entrenar el `VQC`
- ejecutar modo simulador
- ejecutar validación `IBM`
- guardar métricas por número de qubits y por fuente de datos

Entradas posibles:

- `data/dataset.csv`
- `results/live_training_dataset.csv`
- un CSV explícito pasado por argumento

### 4. `src/live_detection`

Responsabilidad:

- capturar tráfico por ventanas
- resumir paquetes en features agregadas
- verificar compatibilidad con el modelo clásico actual

Importante:

- este módulo no lanza ataques
- no opera sobre paquetes individuales a nivel de inferencia
- trabaja con ventanas agregadas

### 5. `app/dashboard`

Responsabilidad:

- mantener el dashboard separado por módulos
- evitar que `app/app.py` concentre toda la lógica

Submódulos:

- `constants.py`: rutas, constantes, colores, metadata base
- `theme.py`: configuración de página y CSS
- `data.py`: carga de resultados y artefactos
- `analytics.py`: gráficos, evaluación clásica, monitoreo live y helpers analíticos
- `ui.py`: componentes visuales simples
- `views.py`: vistas principales del dashboard

## Flujo clásico

```text
data/dataset.csv
    -> train_model.py
    -> artefactos en results/
    -> dashboard
```

El dashboard puede:

- mostrar métricas ya entrenadas
- evaluar `data/dataset.csv`
- evaluar un CSV subido por el usuario

## Flujo cuántico CICIDS

```text
data/dataset.csv
    -> prepare_quantum_dataset()
    -> train_vqc_simulator.py
    -> results/quantum_simulated_metrics*.json
    -> dashboard
```

Modos de ejecución:

- `simulator`
- `ibm_validate`
- `ibm_quantum` (más costoso y dependiente de cuota)

## Flujo cuántico live

```text
captura / pcap
    -> capture.py
    -> feature_extractor.py
    -> results/live_training_dataset.csv
    -> train_vqc_simulator.py --dataset-source live
    -> dashboard
```

El punto clave es que el `VQC live` solo es válido si el modelo fue entrenado con las mismas features que genera la captura live.

## Flujo de monitoreo live en dashboard

El dashboard puede automatizar la captura por lotes para el flujo cuántico `live`.

Secuencia:

1. capturar varias ventanas desde la UI
2. guardar `results/live_capture.csv`
3. opcionalmente anexar al dataset de entrenamiento `results/live_training_dataset.csv`
4. si el dataset live ya es suficiente, entrenar un `VQC` en simulador y predecir ese lote

Límites:

- requiere permisos de captura (`Scapy`)
- no reemplaza una arquitectura de inferencia online persistente
- sigue siendo un laboratorio experimental

## Decisiones de arquitectura actuales

### Separar dashboard de lógica de dominio

La lógica de entrenamiento y captura vive en `src/`.
El dashboard consume esas capacidades y muestra resultados.

### Mantener `results/` como frontera simple

Los modelos y métricas se persisten como archivos.
Eso hace que:

- el dashboard pueda levantarse sin reentrenar
- las corridas sean inspeccionables
- el proyecto siga siendo fácil de mover

### No mezclar ataque y dashboard

El generador de tráfico está deliberadamente fuera del dashboard.
La UI trabaja sobre captura, features, entrenamiento y análisis.

## Deuda técnica restante

Todavía conviene seguir refactorizando:

- dividir `views.py` por pantalla (`overview`, `lab`, `analysis`, etc.)
- unificar tipos de retorno con `dataclass` o `TypedDict`
- limpiar imports duplicados o sobrantes en `src/`
- documentar contratos de `results/*.json`
- separar mejor el flujo de inferencia live del flujo de entrenamiento live

## Criterio de organización futura

Si el proyecto sigue creciendo, la dirección recomendada es:

```text
app/dashboard/views/
    overview.py
    lab.py
    analysis.py
    demo.py
    conclusions.py

src/live_detection/
    capture.py
    feature_extractor.py
    prediction.py
    dataset_builder.py
```

Eso permitiría mantener cada archivo más pequeño, con una responsabilidad bien definida y menos costo de lectura.
