from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_LIVE_FEATURE_COLUMNS = (
    "duration_seconds",
    "total_packets",
    "tcp_packets",
    "udp_packets",
    "icmp_packets",
    "syn_packets",
    "unique_src_ips",
    "unique_dst_ports",
    "packet_rate",
    "avg_packet_size",
)

SCENARIO_LABEL_HINTS = {
    "tcp syn flood avanzado": 1,
    "udp flood con payload variable": 1,
    "icmp flood": 1,
    "ataque hibrido": 1,
    "ataques paralelos": 1,
    "generar trafico de fondo": 0,
}


@dataclass
class LiveCurationReport:
    original_rows: int
    curated_rows: int
    dropped_rows: int
    benign_kept: int
    attack_kept: int
    confidence_threshold: float
    applied: bool
    metadata_dropped_rows: int = 0
    inconsistent_label_rows: int = 0
    legacy_version_rows: int = 0
    fallback_reason: str | None = None


def looks_like_live_feature_frame(columns: list[str] | tuple[str, ...]) -> bool:
    return all(column in set(columns) for column in BASE_LIVE_FEATURE_COLUMNS)


def _normalize_text(value: object) -> str:
    return str(value).strip().lower()


def _build_metadata_keep_mask(metadata: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, int, int]:
    keep_mask = np.ones(len(metadata), dtype=bool)
    inconsistent_label_rows = 0
    legacy_version_rows = 0

    if "Scenario" in metadata.columns:
        scenario_series = metadata["Scenario"].fillna("").map(_normalize_text)
        expected_labels = scenario_series.map(SCENARIO_LABEL_HINTS).to_numpy()
        known_mask = ~pd.isna(expected_labels)
        expected_labels = np.nan_to_num(expected_labels, nan=-1).astype(int)
        inconsistent_mask = known_mask & (expected_labels != y)
        inconsistent_label_rows = int(inconsistent_mask.sum())
        keep_mask &= ~inconsistent_mask

    if "SimulatorVersion" in metadata.columns:
        version_series = metadata["SimulatorVersion"].fillna("").map(_normalize_text)
        known_version_mask = version_series != ""
        legacy_mask = known_version_mask & ~version_series.str.contains("v2", regex=False).to_numpy()
        legacy_version_rows = int(legacy_mask.sum())
        keep_mask &= ~legacy_mask

    return keep_mask, inconsistent_label_rows, legacy_version_rows


def enrich_live_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not looks_like_live_feature_frame(df.columns.tolist()):
        return df.copy()

    enriched = df.copy()
    total_packets = enriched["total_packets"].replace(0, 1)
    tcp_packets = enriched["tcp_packets"].replace(0, 1)
    duration_seconds = enriched["duration_seconds"].replace(0, 1e-9)
    unique_src_ips = enriched["unique_src_ips"].replace(0, 1)
    unique_dst_ports = enriched["unique_dst_ports"].replace(0, 1)

    enriched["tcp_ratio"] = enriched["tcp_packets"] / total_packets
    enriched["udp_ratio"] = enriched["udp_packets"] / total_packets
    enriched["icmp_ratio"] = enriched["icmp_packets"] / total_packets
    enriched["syn_tcp_ratio"] = enriched["syn_packets"] / tcp_packets
    enriched["syn_per_second"] = enriched["syn_packets"] / duration_seconds
    enriched["bytes_per_second"] = enriched["avg_packet_size"] * enriched["packet_rate"]
    enriched["bytes_per_window"] = enriched["avg_packet_size"] * enriched["total_packets"]
    enriched["packets_per_src_ip"] = enriched["total_packets"] / unique_src_ips
    enriched["packets_per_dst_port"] = enriched["total_packets"] / unique_dst_ports
    enriched["ports_per_src_ip"] = enriched["unique_dst_ports"] / unique_src_ips
    enriched["tcp_udp_gap"] = enriched["tcp_packets"] - enriched["udp_packets"]
    enriched["transport_diversity"] = (
        (enriched["tcp_packets"] > 0).astype(int)
        + (enriched["udp_packets"] > 0).astype(int)
        + (enriched["icmp_packets"] > 0).astype(int)
    )
    optional_columns = set(enriched.columns)
    if "ack_packets" in optional_columns:
        enriched["ack_ratio"] = enriched["ack_packets"] / total_packets
    if "rst_packets" in optional_columns:
        enriched["rst_ratio"] = enriched["rst_packets"] / total_packets
    if "fin_packets" in optional_columns:
        enriched["fin_ratio"] = enriched["fin_packets"] / total_packets
    if "psh_packets" in optional_columns:
        enriched["psh_ratio"] = enriched["psh_packets"] / total_packets
    if "tcp_option_packets" in optional_columns:
        enriched["tcp_option_ratio"] = enriched["tcp_option_packets"] / total_packets
    if "packets_with_payload" in optional_columns:
        enriched["payload_ratio"] = enriched["packets_with_payload"] / total_packets
    if "http_ports_packets" in optional_columns:
        enriched["http_ratio"] = enriched["http_ports_packets"] / total_packets
    if "dns_port_packets" in optional_columns:
        enriched["dns_ratio"] = enriched["dns_port_packets"] / total_packets
    if "ntp_port_packets" in optional_columns:
        enriched["ntp_ratio"] = enriched["ntp_port_packets"] / total_packets
    if "snmp_port_packets" in optional_columns:
        enriched["snmp_ratio"] = enriched["snmp_port_packets"] / total_packets
    if "ssdp_port_packets" in optional_columns:
        enriched["ssdp_ratio"] = enriched["ssdp_port_packets"] / total_packets
    if "traceroute_port_packets" in optional_columns:
        enriched["traceroute_ratio"] = enriched["traceroute_port_packets"] / total_packets
    if "avg_ttl" in optional_columns and "std_ttl" in optional_columns:
        enriched["ttl_stability"] = 1.0 / (1.0 + enriched["std_ttl"])
    return enriched


def curate_live_windows(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    metadata: pd.DataFrame | None = None,
    confidence_threshold: float = 0.5,
    random_state: int = 42,
    minimum_rows_per_class: int = 8,
) -> tuple[pd.DataFrame, np.ndarray, LiveCurationReport]:
    y = np.asarray(y)
    original_rows = len(X)
    metadata_dropped_rows = 0
    inconsistent_label_rows = 0
    legacy_version_rows = 0
    if metadata is not None and not metadata.empty:
        metadata = metadata.reset_index(drop=True)
        metadata_mask, inconsistent_label_rows, legacy_version_rows = _build_metadata_keep_mask(metadata, y)
        metadata_dropped_rows = int((~metadata_mask).sum())
        if metadata_dropped_rows:
            X = X.loc[metadata_mask].reset_index(drop=True)
            y = y[metadata_mask]

    if len(X) < 20 or len(np.unique(y)) < 2:
        report = LiveCurationReport(
            original_rows=original_rows,
            curated_rows=len(X),
            dropped_rows=original_rows - len(X),
            benign_kept=int((y == 0).sum()),
            attack_kept=int((y == 1).sum()),
            confidence_threshold=confidence_threshold,
            applied=False,
            metadata_dropped_rows=metadata_dropped_rows,
            inconsistent_label_rows=inconsistent_label_rows,
            legacy_version_rows=legacy_version_rows,
            fallback_reason="dataset_too_small",
        )
        return X, y, report

    max_supported_splits = min(5, int((y == 1).sum()), int((y == 0).sum()))
    if max_supported_splits < 2:
        report = LiveCurationReport(
            original_rows=original_rows,
            curated_rows=len(X),
            dropped_rows=original_rows - len(X),
            benign_kept=int((y == 0).sum()),
            attack_kept=int((y == 1).sum()),
            confidence_threshold=confidence_threshold,
            applied=False,
            metadata_dropped_rows=metadata_dropped_rows,
            inconsistent_label_rows=inconsistent_label_rows,
            legacy_version_rows=legacy_version_rows,
            fallback_reason="insufficient_class_support",
        )
        return X, y, report
    splitter = StratifiedKFold(n_splits=max_supported_splits, shuffle=True, random_state=random_state)

    probabilities = np.zeros(len(X), dtype=float)
    for train_idx, eval_idx in splitter.split(X, y):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        probabilities[eval_idx] = model.predict_proba(X.iloc[eval_idx])[:, 1]

    confidence = np.where(y == 1, probabilities, 1.0 - probabilities)
    keep_mask = confidence >= confidence_threshold
    curated_y = y[keep_mask]

    benign_kept = int((curated_y == 0).sum())
    attack_kept = int((curated_y == 1).sum())
    if benign_kept < minimum_rows_per_class or attack_kept < minimum_rows_per_class:
        report = LiveCurationReport(
            original_rows=original_rows,
            curated_rows=len(X),
            dropped_rows=original_rows - len(X),
            benign_kept=int((y == 0).sum()),
            attack_kept=int((y == 1).sum()),
            confidence_threshold=confidence_threshold,
            applied=False,
            metadata_dropped_rows=metadata_dropped_rows,
            inconsistent_label_rows=inconsistent_label_rows,
            legacy_version_rows=legacy_version_rows,
            fallback_reason="curation_too_aggressive",
        )
        return X, y, report

    curated_X = X.loc[keep_mask].reset_index(drop=True)
    curated_y = curated_y.copy()
    report = LiveCurationReport(
        original_rows=original_rows,
        curated_rows=len(curated_X),
        dropped_rows=original_rows - len(curated_X),
        benign_kept=benign_kept,
        attack_kept=attack_kept,
        confidence_threshold=confidence_threshold,
        applied=True,
        metadata_dropped_rows=metadata_dropped_rows,
        inconsistent_label_rows=inconsistent_label_rows,
        legacy_version_rows=legacy_version_rows,
    )
    return curated_X, curated_y, report
