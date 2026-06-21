# Live Detection Experimental

Este modulo crea un puente experimental entre trafico capturado en vivo con Scapy y los modelos del proyecto.

## Objetivo

- Capturar trafico durante una ventana de tiempo.
- Resumir la captura en un conjunto pequeno de features agregadas.
- Guardar una fila en CSV para futuras pruebas de inferencia.
- Validar si esas columnas son compatibles con el modelo clasico actual.
- Dejar documentado que el VQC necesita reentrenamiento con exactamente las mismas features en vivo.

## Archivos

- `capture.py`: CLI para capturar paquetes y generar `live_capture.csv`.
- `feature_extractor.py`: logica de extraccion de features agregadas.
- `predict_live.py`: valida compatibilidad e intenta inferencia clasica solo si las columnas coinciden.

## Features extraidas

La captura actual genera una fila con estas columnas:

- `duration_seconds`
- `total_packets`
- `tcp_packets`
- `udp_packets`
- `icmp_packets`
- `syn_packets`
- `unique_src_ips`
- `unique_dst_ports`
- `packet_rate`
- `avg_packet_size`

## Uso

Desde la raiz del proyecto:

```bash
python -m src.live_detection.capture --duration 10 --output results/live_capture.csv
```

Si el entorno no expone `python`, usa:

```bash
python3 -m src.live_detection.capture --duration 10 --output results/live_capture.csv
```

Si no tenes permisos para captura en vivo, podes extraer features desde un archivo `.pcap` ya generado:

```bash
python3 -m src.live_detection.capture --duration 1 --pcap ddos_test.pcap --output results/live_capture.csv
```

Para validar compatibilidad y, solo si aplica, predecir con el modelo clasico:

```bash
python -m src.live_detection.predict_live --input results/live_capture.csv
```

Para construir un dataset supervisado de trafico en vivo y luego entrenar el VQC experimental, podes etiquetar y acumular capturas:

```bash
python3 -m src.live_detection.capture --duration 10 --output results/live_training_dataset.csv --label benign --append
python3 -m src.live_detection.capture --duration 10 --output results/live_training_dataset.csv --label attack --append
```

Si queres construir un dataset mas grande sin ejecutar el comando muchas veces, usa `--windows`:

```bash
python3 -m src.live_detection.capture --duration 2 --windows 100 --output results/live_training_dataset.csv --label benign --append
python3 -m src.live_detection.capture --duration 2 --windows 100 --output results/live_training_dataset.csv --label attack --append
```

Eso genera 200 filas totales: 100 benign y 100 attack.

Si usas PCAPs en vez de sniff en vivo, el mismo flujo funciona cambiando `--duration ...` por `--pcap archivo.pcap`.

## Compatibilidad con el modelo actual

El modelo clasico entrenado hoy usa features numericas de `CICIDS2017` tomadas desde `data/dataset.csv`, luego aplica `StandardScaler` y `PCA`.

Eso significa que `live_capture.csv` no debe usarse para inferencia salvo que sus columnas coincidan exactamente con las usadas durante entrenamiento.

Si no coinciden, `predict_live.py` informa:

`El modelo actual fue entrenado con features CICIDS2017. Para usar live_capture.csv se debe entrenar un modelo nuevo con estas mismas features.`

## Nota sobre VQC

El VQC simulado tambien depende por completo del espacio de features de entrenamiento. Si se quiere inferencia cuantica sobre datos en vivo, primero hay que reentrenar el pipeline cuantico con este mismo conjunto de features extraidas en vivo.
