<div align="center">

# Quantum IDS

### Detección de intrusiones con Machine Learning clásico y computación cuántica

![Azul Boca](https://img.shields.io/badge/Identidad-Azul%20Boca-003B7A?style=for-the-badge)
![Oro Boca](https://img.shields.io/badge/Contraste-Oro%20Boca-FDB913?style=for-the-badge&labelColor=003B7A)
![Python](https://img.shields.io/badge/Python-3.10+-003B7A?style=for-the-badge&logo=python&logoColor=FDB913)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FDB913?style=for-the-badge&logo=streamlit&logoColor=003B7A)

**Tesina de Licenciatura en Sistemas**<br>
Ticiana Angelucci · Universidad Champagnat · 2026

</div>

---

## Descripción

Quantum IDS es una plataforma experimental para comparar estrategias de detección de intrusiones en redes mediante:

- un baseline clásico basado en **Random Forest**;
- un **Quantum Support Vector Machine (QSVM)** con kernel de fidelidad;
- simulación local mediante **Qiskit**;
- validación física sobre una **SpinQ Triangulum de 3 qubits**, conectada mediante SpinQuasar;
- captura y evaluación de tráfico en vivo por ventanas.

La investigación utiliza CICIDS2017 como conjunto de referencia y permite contrastar rendimiento, estabilidad, costo computacional y efecto del ruido al pasar del entorno ideal al hardware cuántico real.

> [!IMPORTANT]
> El objetivo no es asumir una ventaja cuántica inmediata. La contribución consiste en construir un pipeline reproducible para evaluar los enfoques clásico y cuántico bajo muestras, métricas y condiciones controladas.

## Arquitectura experimental

```text
Tráfico CICIDS2017 o Live
            |
            v
Limpieza, balanceo y selección de características
            |
            +-----------------------------+
            |                             |
            v                             v
 Random Forest                  Codificación en 3-N qubits
  (baseline)                              |
                                         v
                              Kernel cuántico de fidelidad
                                /                 \
                               v                   v
                      Simulador Qiskit       SpinQ Triangulum
                                \                 /
                                 v               v
                                  SVC precomputado
                                         |
                                         v
                    Accuracy · Precision · Recall · F1
```

El hardware SpinQ calcula similitudes entre pares de muestras. Esas fidelidades forman las matrices de entrenamiento y prueba utilizadas por una SVM clásica con `kernel="precomputed"`.

## Funcionalidades del dashboard

La interfaz mantiene una identidad visual inspirada en los colores de Boca: azul profundo, blanco y oro.

### 1. Resumen

- Presentación de la hipótesis.
- Contraste entre Random Forest, QSVM y hardware real.
- Comparación de métricas y paradigmas.

### 2. Experimentar

- Evaluación del baseline clásico.
- QSVM sobre CICIDS2017.
- Simulador local Qiskit.
- Prueba de conectividad SpinQ de un circuito.
- QSVM piloto SpinQ de siete circuitos.
- Contador y progreso de las ejecuciones físicas.

### 3. Live

- Captura de tráfico por ventanas.
- Simuladores de laboratorio v2 y v3 multivectorial.
- Escenarios benignos y de ataque.
- Construcción de un dataset Live balanceado.
- QSVM Live en simulador local o SpinQ.

### 4. Análisis y Síntesis

- Rendimiento comparado.
- Corridas cuánticas disponibles.
- Ruido y limitaciones del hardware.
- Costos temporales.
- Hallazgos y conclusión integradora de la tesina.

## QSVM sobre SpinQ

La SpinQ Triangulum utiliza exactamente **3 qubits**. Al seleccionar SpinQ, el dashboard fija automáticamente esta dimensionalidad en toda la interfaz.

Cada circuito compara dos muestras mediante:

```text
U(x_a) -> U†(x_b) -> medición
```

La fidelidad se estima con la probabilidad de medir `000`:

```text
fidelidad = counts["000"] / shots_totales
```

### Modos disponibles

| Modo | Circuitos | Shots por circuito | Propósito |
|---|---:|---:|---|
| Conectividad | 1 | 1024 | Verificar la comunicación con SpinQuasar |
| QSVM piloto | 7 | 1024 | Validar el pipeline completo con datos reales |

> [!WARNING]
> El piloto utiliza 2 muestras de entrenamiento y 2 de prueba. Sirve para validar integración, construcción del kernel y clasificación, pero sus métricas no son estadísticamente representativas. Por ejemplo, con dos registros de prueba la accuracy sólo puede cambiar en saltos de 50 puntos porcentuales.

## Estructura del proyecto

```text
app/
  app.py                         # Entrada principal de Streamlit
  dashboard/
    analytics.py                 # Métricas, evaluación y gráficos
    data.py                      # Carga de resultados y modelos
    theme.py                     # Tema azul y oro
    ui.py                        # Componentes compartidos
    pages/
      overview.py                # Resumen académico
      lab.py                     # Laboratorio clásico y QSVM
      live.py                    # Captura y experimentación Live
      analysis.py                # Análisis y Síntesis

src/
  classical/                     # Random Forest y baseline Live
  live_detection/                # Captura y extracción de features
  preprocessing/                 # Preparación del dataset cuántico
  quantum/
    spinq_connector.py           # Configuración SpinQ/SpinQuasar
    train_qkernel.py             # Entrenamiento del Quantum Kernel
    train_vqc_simulator.py       # Flujo VQC experimental heredado
    runtime.py                   # Integraciones de runtime cuántico

scripts/
  01_attack-scrapy_v2.py         # Simulador avanzado de tráfico
  01_attack-scrapy_v3.py         # Simulador multivectorial

data/                            # Dataset local, fuera de Git
results/                         # Modelos y métricas, fuera de Git
docs/                            # Documentación técnica
```

## Instalación

### Requisitos

- Python 3.10 o superior.
- `pip` y soporte para entornos virtuales.
- Dataset disponible en `data/dataset.csv`.
- Para hardware real: Windows, SpinQit, acceso de red a SpinQuasar y credenciales autorizadas.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app/app.py
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux y macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app/app.py
```

El dashboard queda disponible normalmente en:

```text
http://localhost:8501
```

Para exponerlo en la red local:

```bash
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

## Configuración de SpinQ

SpinQit no forma parte del flujo obligatorio del simulador. Debe instalarse en el mismo entorno virtual utilizado para ejecutar Streamlit:

```powershell
python -m pip install spinqit
```

Antes de una prueba física:

1. Iniciar SpinQuasar.
2. Confirmar conectividad de red con el equipo.
3. Configurar IP, puerto y cuenta de manera local.
4. Seleccionar `Hardware Real SpinQ` en el dashboard.
5. Ejecutar primero la prueba de conectividad de un circuito.

> [!CAUTION]
> No publiques IPs privadas, usuarios ni contraseñas del laboratorio. Las credenciales deben gestionarse localmente y quedar fuera del control de versiones.

## Datos y resultados

### Dataset principal

```text
data/dataset.csv
```

### Artefactos clásicos

```text
results/random_forest_model.joblib
results/scaler.joblib
results/pca.joblib
results/classical_metrics.json
```

### Dataset Live

```text
results/live_training_dataset.csv
```

Las carpetas `data/` y `results/` están excluidas de Git porque pueden contener datasets grandes, capturas o artefactos generados localmente.

## Métricas

| Métrica | Interpretación en el IDS |
|---|---|
| Accuracy | Proporción total de decisiones correctas |
| Precision | Confiabilidad de las alertas de ataque |
| Recall | Capacidad para detectar ataques reales |
| F1-Score | Equilibrio entre Precision y Recall |

En ciberseguridad no conviene interpretar Accuracy de manera aislada. Un modelo que clasifica todo como benigno puede obtener aciertos y, al mismo tiempo, presentar `Recall = 0` para la clase de ataque.

## Documentación adicional

- [Arquitectura general](docs/ARCHITECTURE.md)
- [Captura y procesamiento Live](src/live_detection/README.md)
- [Flujo cuántico](src/quantum/README.md)

---

<div align="center">

**Quantum IDS** · Ciberseguridad & Computación Cuántica<br>
<sub>Azul para la arquitectura. Oro para la evidencia.</sub>

</div>
