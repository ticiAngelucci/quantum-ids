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
    pages/                # pantallas del dashboard separadas por responsabilidad

src/
  classical/              # entrenamiento y utilidades del modelo clásico
  live_detection/         # captura live, extracción, curación y compatibilidad
  preprocessing/          # limpieza, escalado y PCA para QML
  quantum/                # configuración, runtime, resultados y entrenamiento VQC
  utils/                  # helpers compartidos
```

## Puesta en marcha

### Requisitos

- Python `3.10` o superior
- `pip`
- acceso a una terminal
- opcional: entorno virtual `venv`

### Ubuntu

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/app.py
```

### macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/app.py
```

Si `python3` no existe, primero instalá Python con Homebrew o desde el instalador oficial.

### Windows PowerShell

```powershell
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/app.py
```

Si PowerShell bloquea la activación del entorno virtual, podés habilitarla para tu usuario con:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows CMD

```cmd
py -3 -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/app.py
```

### URL del dashboard

Cuando Streamlit arranca, normalmente queda disponible en:

- `http://localhost:8501`
- o `http://IP_LOCAL:8501` si lo levantás para verlo desde otra máquina de la red

Ejemplo:

```bash
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
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
  - `results/quantum_simulated_metrics_3q.json`
  - `results/quantum_simulated_metrics_4q.json`
  - `results/quantum_simulated_metrics_6q.json`
  - `results/quantum_simulated_metrics_8q.json`

### 3. Cuántico live

- captura por ventanas: `src/live_detection/capture.py`
- extracción de features: `src/live_detection/feature_extractor.py`
- enriquecimiento y curación: `src/live_detection/feature_engineering.py`
- simulador asumido por la UI: `v2`
- metadata opcional de captura:
  - `Scenario`
  - `SimulatorVersion`
- dataset experimental: `results/live_training_dataset.csv`
- baseline comparativo sobre live:
  - `python -m src.classical.train_live_model`
- resultados clásicos live:
  - `results/classical_live_metrics.json`
- entrenamiento cuántico live:
  - `python -m src.quantum.train_vqc_simulator --dataset-source live --qubits 2`

Si querés aprovechar las mejoras más recientes del flujo `live v2`, conviene reconstruir el CSV desde cero:

```bash
rm results/live_training_dataset.csv
```

Después recapturá ventanas nuevas. Las capturas nuevas incluyen features más discriminativas del tráfico `v2` y permiten curar filas inconsistentes según el escenario del simulador.

## Configuración del VQC

El pipeline cuántico hoy se puede ajustar sin tocar código, tanto por CLI como desde el dashboard:

- `qubits`: `2`, `3`, `4`, `6`, `8`
- `feature_map_reps`
- `ansatz_reps`
- `maxiter` de `COBYLA`
- `test_size`

Ejemplo:

```bash
python -m src.quantum.train_vqc_simulator \
  --qubits 4 \
  --feature-map-reps 1 \
  --ansatz-reps 2 \
  --maxiter 100 \
  --test-size 0.2
```

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
