# quantum-ids

`quantum-ids` es un proyecto experimental de detección de anomalías en tráfico de red que compara:

- un baseline clásico con `Random Forest`
- un enfoque de `Quantum Machine Learning` con `VQC`
- una validación opcional sobre `IBM Quantum`

El proyecto también incluye un flujo `live` para capturar tráfico en ventanas, extraer features agregadas y usarlas como dataset experimental para el modelo cuántico.

## Estructura

```text
app/
  app.py                  # entrypoint del dashboard Streamlit
  dashboard/              # UI modular, gráficos, carga de datos y vistas

src/
  classical/              # entrenamiento y utilidades del modelo clásico
  live_detection/         # captura live, extracción de features y compatibilidad
  preprocessing/          # limpieza, escalado y PCA para QML
  quantum/                # entrenamiento VQC y validación IBM
  utils/                  # helpers compartidos
```

## Flujos principales

### 1. Clásico

- dataset base: `data/dataset.csv`
- entrenamiento: `src/classical/train_model.py`
- resultados esperados:
  - `results/classical_metrics.json`
  - `results/random_forest_model.joblib`
  - `results/scaler.joblib`
  - `results/pca.joblib`

### 2. Cuántico sobre CICIDS

- entrenamiento: `src/quantum/train_vqc_simulator.py`
- fuente de datos: `data/dataset.csv`
- resultados esperados:
  - `results/quantum_simulated_metrics.json`
  - `results/quantum_simulated_metrics_2q.json`
  - `results/quantum_simulated_metrics_4q.json`
  - `results/quantum_simulated_metrics_6q.json`
  - `results/quantum_simulated_metrics_8q.json`

### 3. Cuántico live

- captura por ventanas: `src/live_detection/capture.py`
- extracción de features: `src/live_detection/feature_extractor.py`
- dataset experimental: `results/live_training_dataset.csv`
- entrenamiento cuántico live:
  - `python -m src.quantum.train_vqc_simulator --dataset-source live --qubits 2`

## Dashboard

El dashboard corre con:

```bash
streamlit run app/app.py
```

Responsabilidades del dashboard:

- cargar métricas reales si existen en `results/`
- comparar clásico, VQC simulado y hardware real
- ejecutar pruebas del laboratorio
- operar el flujo `live` cuántico desde la UI

## Documentación

- arquitectura general: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- flujo live: [src/live_detection/README.md](src/live_detection/README.md)
- flujo cuántico: [src/quantum/README.md](src/quantum/README.md)
