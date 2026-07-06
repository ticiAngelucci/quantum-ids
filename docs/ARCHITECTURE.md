# Arquitectura

## Objetivo

`quantum-ids` compara un pipeline clásico de detección de anomalías contra un pipeline de `Quantum Machine Learning`, manteniendo además un flujo experimental `live` para trabajar con tráfico capturado en laboratorio.

La arquitectura actual separa cuatro responsabilidades principales:

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
          ├── types.py
          ├── ui.py
          ├── views.py          # fachada mínima que reexporta vistas
          └── pages/*

src/live_detection/
    ├── capture.py
    ├── feature_extractor.py
    ├── feature_engineering.py
    ├── compatibility.py        # compatibilidad entre features live y modelo clásico
    └── predict_live.py
```

## Estructura actual resumida

```text
app/
  app.py
  dashboard/
    analytics.py
    constants.py
    data.py
    theme.py
    types.py
    ui.py
    views.py
    pages/
      overview.py
      lab.py
      analysis.py
      demo.py
      conclusions.py

src/
  classical/
  live_detection/
  preprocessing/
  quantum/
  utils/
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
- separar runtime, configuración y resultados en módulos pequeños

Entradas posibles:

- `data/dataset.csv`
- `results/live_training_dataset.csv`
- un CSV explícito pasado por argumento

Submódulos relevantes:

- `config.py`: defaults y contratos de ejecución cuántica
- `runtime.py`: construcción del `VQC`, imports Qiskit y utilidades IBM
- `results.py`: métricas, paths y helpers de validación
- `train_vqc_simulator.py`: orquestador y CLI

### 4. `src/live_detection`

Responsabilidad:

- capturar tráfico por ventanas
- resumir paquetes en features agregadas
- verificar compatibilidad con el modelo clásico actual
- reutilizar esa validación tanto en CLI como en dashboard

Importante:

- este módulo no lanza ataques
- no opera sobre paquetes individuales a nivel de inferencia
- trabaja con ventanas agregadas

Módulos relevantes:

- `capture.py`: CLI para captura live o lectura de PCAP
- `feature_extractor.py`: cálculo de features agregadas por ventana
- `feature_engineering.py`: enriquecimiento de features y curación de ventanas ambiguas
- `compatibility.py`: contrato de columnas esperado por el baseline clásico
- `predict_live.py`: validación de compatibilidad e inferencia clásica solo si las columnas coinciden

### 5. `app/dashboard`

Responsabilidad:

- mantener el dashboard separado por módulos
- evitar que `app/app.py` concentre toda la lógica
- diferenciar claramente configuración, datos, componentes visuales y páginas

Submódulos:

- `constants.py`: rutas, constantes, colores, metadata base
- `theme.py`: configuración de página y CSS
- `data.py`: carga de resultados y artefactos
- `types.py`: contratos compartidos del dashboard
- `analytics.py`: gráficos, evaluación clásica, monitoreo live y helpers analíticos
- `ui.py`: componentes visuales simples
- `views.py`: capa de compatibilidad que reexporta las páginas
- `pages/`: vistas principales del dashboard separadas por pantalla

Páginas actuales:

- `overview.py`: lectura general del estado del sistema
- `lab.py`: laboratorio clásico, laboratorio cuántico y monitoreo live
- `analysis.py`: comparación, corridas VQC, ruido y tiempos
- `demo.py`: simulación simple orientada a explicación
- `conclusions.py`: cierre ejecutivo del dashboard

La UI hoy también expone ajustes del circuito cuántico:

- `feature_map_reps`
- `ansatz_reps`
- `optimizer_maxiter`

Esto permite experimentar sin tocar código ni dejar parámetros importantes escondidos en el backend.

## Contratos importantes

### 1. `results/` como frontera de integración

El proyecto usa `results/` como frontera simple entre entrenamiento, captura y visualización.

Archivos clave:

- `results/classical_metrics.json`
- `results/classical_live_metrics.json`
- `results/random_forest_model.joblib`
- `results/scaler.joblib`
- `results/pca.joblib`
- `results/quantum_simulated_metrics*.json`
- `results/quantum_hardware_metrics*.json`
- `results/quantum_live_simulated_metrics*.json`
- `results/quantum_live_hardware_metrics*.json`
- `results/live_capture.csv`
- `results/live_training_dataset.csv`

### 2. `model_data` en dashboard

`app/dashboard/data.py` arma un `model_data` unificado para:

- `Modelo clasico`
- `Modelo cuantico`
- `Hardware cuantico real`

Ese contrato es consumido por:

- páginas
- gráficos
- sidebar
- tarjetas de resumen

### 3. `SidebarSelection`

El estado principal de navegación del dashboard se sintetiza en `SidebarSelection`:

- modelo activo
- qubits activos
- fuente cuántica activa
- sección activa

Eso evita que la UI y el renderizado queden desincronizados por cambios de `session_state`.

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

Parámetros de circuito hoy visibles en CLI/UI:

- cantidad de qubits / componentes PCA
- repeticiones del feature map
- repeticiones del ansatz
- iteraciones máximas de `COBYLA`

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

El flujo actual de `live` incluye una etapa extra antes del VQC:

1. enriquecimiento de features agregadas
2. curación de ventanas ambiguas con un proxy clásico out-of-fold
3. escalado + PCA
4. entrenamiento/evaluación del VQC

En otras palabras:

- captura live != detección inmediata
- captura live -> dataset live -> entrenamiento/evaluación VQC live

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

Esto permite:

- probar scripts por terminal sin depender de Streamlit
- mantener la UI como capa de orquestación
- desacoplar el laboratorio visual del pipeline de entrenamiento

### Mantener `results/` como frontera simple

Los modelos y métricas se persisten como archivos.
Eso hace que:

- el dashboard pueda levantarse sin reentrenar
- las corridas sean inspeccionables
- el proyecto siga siendo fácil de mover

### No mezclar ataque y dashboard

El generador de tráfico está deliberadamente fuera del dashboard.
La UI trabaja sobre captura, features, entrenamiento y análisis.

### Introducir tipado liviano en la UI

El dashboard ya no depende solo de `dict` anónimos:

- `types.py` define contratos para selección de sidebar y estructura de `model_data`
- esto reduce errores de sincronización y hace más legibles los cambios futuros

## Deuda técnica restante

Todavía conviene seguir refactorizando:

- partir `app/dashboard/pages/lab.py`, que hoy sigue siendo la pantalla más grande
- partir `app/dashboard/pages/live.py`, que sigue acumulando demasiada lógica de UI
- expandir el tipado mas alla del dashboard hacia `src/`
- limpiar imports duplicados o sobrantes en `src/`
- documentar contratos de `results/*.json`
- separar mejor el flujo de inferencia live del flujo de entrenamiento live
- evaluar mover persistencia de sesión del dashboard a helpers dedicados

## Criterio de organización futura

Si el proyecto sigue creciendo, la dirección recomendada es:

```text
app/dashboard/pages/
    overview.py
    analysis.py
    demo.py
    conclusions.py
    lab/
      classical.py
      quantum.py
      live_monitor.py

src/live_detection/
    capture.py
    feature_extractor.py
    compatibility.py
    prediction.py
    dataset_builder.py
```

Eso permitiría mantener cada archivo más pequeño, con una responsabilidad bien definida y menos costo de lectura, especialmente en el laboratorio del dashboard.
