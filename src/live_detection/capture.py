from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src.live_detection.feature_extractor import extract_live_features


LOGGER = logging.getLogger("live_detection.capture")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura trafico en vivo por una ventana de tiempo y resume features experimentales."
    )
    parser.add_argument("--duration", type=int, required=True, help="Segundos de captura.")
    parser.add_argument("--output", type=Path, required=True, help="CSV de salida.")
    parser.add_argument("--iface", type=str, default=None, help="Interfaz de red opcional para Scapy.")
    parser.add_argument(
        "--pcap",
        type=Path,
        default=None,
        help="Ruta opcional a un archivo PCAP para extraer features sin captura en vivo.",
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=("benign", "attack"),
        default=None,
        help="Etiqueta opcional para construir un dataset supervisado de trafico en vivo.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Agrega la fila al CSV existente en lugar de sobrescribirlo.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Limite opcional de paquetes a capturar. 0 significa sin limite adicional.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=1,
        help="Cantidad de ventanas independientes a capturar y guardar como filas separadas.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Escenario opcional del simulador para guardar metadata del lote live.",
    )
    parser.add_argument(
        "--simulator-version",
        type=str,
        default=None,
        help="Version opcional del simulador para guardar metadata del lote live.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def capture_packets(duration: int, iface: str | None = None, count: int = 0):
    try:
        from scapy.all import sniff
    except ImportError as error:
        raise ImportError(
            "No se pudo importar Scapy. Instala las dependencias del proyecto para usar captura en vivo."
        ) from error

    LOGGER.info("Iniciando captura por %s segundos%s", duration, f" en interfaz {iface}" if iface else "")
    started_at = time.perf_counter()
    try:
        packets = sniff(timeout=duration, iface=iface, store=True, count=count)
    except PermissionError as error:
        raise PermissionError(
            "Scapy no pudo abrir un socket raw para capturar trafico en vivo. "
            "Ejecuta el comando con permisos elevados o usa --pcap para procesar una captura existente."
        ) from error
    elapsed = time.perf_counter() - started_at
    LOGGER.info("Captura finalizada. Paquetes capturados: %s", len(packets))
    return packets, elapsed


def load_packets_from_pcap(pcap_path: Path):
    try:
        from scapy.all import rdpcap
    except ImportError as error:
        raise ImportError(
            "No se pudo importar Scapy. Instala las dependencias del proyecto para procesar PCAP."
        ) from error

    if not pcap_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo PCAP {pcap_path}")

    LOGGER.info("Cargando paquetes desde PCAP %s", pcap_path)
    started_at = time.perf_counter()
    packets = rdpcap(str(pcap_path))
    elapsed = time.perf_counter() - started_at
    LOGGER.info("PCAP cargado. Paquetes leidos: %s", len(packets))
    return packets, elapsed


def save_features(features: dict[str, float], output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_row_df = pd.DataFrame([features])
    
    if append:
        # mode='a' es "append". Si el archivo no existe, escribe los encabezados. Si existe, no los repite.
        new_row_df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)
    else:
        # mode='w' es "write" (sobrescribe todo desde cero)
        new_row_df.to_csv(output_path, mode='w', header=True, index=False)
        
    LOGGER.info("Features guardadas en %s", output_path)


def main() -> None:
    args = parse_args()
    configure_logging()

    if args.duration <= 0:
        raise ValueError("--duration debe ser mayor a 0.")
    if args.windows <= 0:
        raise ValueError("--windows debe ser mayor a 0.")

    for window_index in range(args.windows):
        LOGGER.info("Procesando ventana %s de %s", window_index + 1, args.windows)
        if args.pcap is not None:
            packets, elapsed = load_packets_from_pcap(args.pcap)
        else:
            packets, elapsed = capture_packets(duration=args.duration, iface=args.iface, count=args.count)

        features = extract_live_features(packets=packets, duration_seconds=elapsed)
        if args.label is not None:
            features["Label"] = args.label
            LOGGER.info("Etiqueta asignada a la captura: %s", args.label)
        if args.scenario is not None:
            features["Scenario"] = args.scenario
        if args.simulator_version is not None:
            features["SimulatorVersion"] = args.simulator_version
        LOGGER.info("Resumen extraido: %s", features)
        save_features(
            features,
            args.output,
            append=args.append or window_index > 0,
        )


if __name__ == "__main__":
    main()
