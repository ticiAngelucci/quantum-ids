# Quantum Training Modes

El VQC ahora soporta dos modos de trabajo separados.

## 1. Modo actual: CICIDS2017

Mantiene el comportamiento original del proyecto. Usa `data/dataset.csv` y guarda resultados en:

- `results/quantum_simulated_metrics.json`
- `results/quantum_simulated_metrics_{N}q.json`

Ejemplo:

```bash
python3 -m src.quantum.train_vqc_simulator --dataset-source cicids --qubits 4
```

## 2. Modo experimental: trafico live del simulador

Este modo entrena el VQC con las features agregadas capturadas en vivo. Requiere primero construir un dataset etiquetado con columnas compatibles y una etiqueta `Label`.

Ruta esperada por defecto:

- `results/live_training_dataset.csv`

Resultados generados:

- `results/quantum_live_simulated_metrics.json`
- `results/quantum_live_simulated_metrics_{N}q.json`

Ejemplo:

```bash
python3 -m src.quantum.train_vqc_simulator --dataset-source live --qubits 4
```

## 3. Modo IBM Quantum real

Este modo mantiene el mismo pipeline de datos, pero ejecuta el `Sampler` sobre hardware remoto de IBM Quantum usando `qiskit-ibm-runtime`.

Antes de usarlo:

```bash
export IBM_QUANTUM_TOKEN="tu_token"
```

Opcionalmente:

```bash
export IBM_QUANTUM_INSTANCE="tu_instancia"
```

### Opcion recomendada: entrenamiento local + validacion IBM

Esta es la modalidad pensada para ahorrar cuota mensual. Entrena el VQC en simulador y envia a IBM solo un subconjunto chico del test para medir impacto de ruido, cola y hardware real.

Ejemplo con CICIDS:

```bash
python3 -m src.quantum.train_vqc_simulator --execution-target ibm_validate --dataset-source cicids --qubits 4 --ibm-validation-samples 16
```

Ejemplo con dataset live:

```bash
python3 -m src.quantum.train_vqc_simulator --execution-target ibm_validate --dataset-source live --qubits 2 --test-size 0.5 --ibm-validation-samples 8
```

### Opcion costosa: entrenamiento completo en IBM

Solo para pruebas puntuales si tenes cuota disponible.

Ejemplo con CICIDS:

```bash
python3 -m src.quantum.train_vqc_simulator --execution-target ibm_quantum --dataset-source cicids --qubits 4
```

Ejemplo con dataset live:

```bash
python3 -m src.quantum.train_vqc_simulator --execution-target ibm_quantum --dataset-source live --qubits 2 --test-size 0.5
```

Resultados generados:

- `results/quantum_hardware_metrics.json`
- `results/quantum_hardware_metrics_{N}q.json`
- `results/quantum_live_hardware_metrics.json`
- `results/quantum_live_hardware_metrics_{N}q.json`

## Metricas de ruido y limitaciones de hardware

Cuando el modo IBM esta activo, el resultado tambien guarda:

- `hardware_diagnostics.backend_name`
- `hardware_diagnostics.pending_jobs`
- `hardware_diagnostics.avg_t1_us`
- `hardware_diagnostics.avg_t2_us`
- `hardware_diagnostics.coupling_edge_count`
- `hardware_diagnostics.limitation_flags`
- `hardware_gap_vs_simulator.accuracy_drop`
- `hardware_gap_vs_simulator.f1_drop`

Esto te sirve para medir impacto de cola, conectividad y decoherencia frente al baseline ideal del simulador.

## Como construir el dataset live

Captura trafico normal y de ataque en ventanas separadas, etiquetando cada fila y acumulandola en el mismo CSV.

Trafico benigno:

```bash
python3 -m src.live_detection.capture --duration 10 --output results/live_training_dataset.csv --label benign --append
```

Trafico de ataque:

```bash
python3 -m src.live_detection.capture --duration 10 --output results/live_training_dataset.csv --label attack --append
```

Despues de varias capturas benignas y de ataque, entrena el VQC live.

Para un experimento mas estable, podes generar 200 filas totales asi:

```bash
python3 -m src.live_detection.capture --duration 2 --windows 100 --output results/live_training_dataset.csv --label benign --append
python3 -m src.live_detection.capture --duration 2 --windows 100 --output results/live_training_dataset.csv --label attack --append
python3 -m src.quantum.train_vqc_simulator --dataset-source live --qubits 4
```

Si no podes capturar en vivo por permisos de Scapy, genera o reutiliza archivos `.pcap` y procesa cada uno con `--pcap`.

## Limite importante

El modo `live` no reutiliza el VQC entrenado con CICIDS2017. Es un segundo pipeline supervisado, porque el espacio de features es distinto.
