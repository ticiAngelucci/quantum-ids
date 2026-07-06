from __future__ import annotations

from pathlib import Path


DATASET_PATH = Path("data/dataset.csv")
RESULTS_DIR = Path("results")
LATEST_RESULTS_PATH = RESULTS_DIR / "quantum_simulated_metrics.json"
HARDWARE_RESULTS_PATH = RESULTS_DIR / "quantum_hardware_metrics.json"
LIVE_DATASET_PATH = RESULTS_DIR / "live_training_dataset.csv"

BENIGN_SAMPLES = 200
ATTACK_SAMPLES = 200
DEFAULT_QUBITS = 4
SUPPORTED_QUBITS = (2, 3, 4, 6, 8)
SUPPORTED_DATASET_SOURCES = ("cicids", "live")
SUPPORTED_EXECUTION_TARGETS = ("simulator", "ibm_quantum", "ibm_validate")

TEST_SIZE = 0.2
RANDOM_STATE = 42
COBYLA_MAXITER = 50
DEFAULT_IBM_SHOTS = 1024
DEFAULT_IBM_VALIDATION_SAMPLES = 16
DEFAULT_FEATURE_MAP_REPS = 1
DEFAULT_ANSATZ_REPS = 1
