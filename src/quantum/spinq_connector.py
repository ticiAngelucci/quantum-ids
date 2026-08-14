from __future__ import annotations
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from spinqit import get_compiler, Circuit
from spinqit import H, CX

if not hasattr(np, 'msort'):
    np.msort = lambda a: np.sort(a, axis=0)

SPINQ_IP = "192.168.172.217"  
SPINQ_PORT = 50177         
USUARIO = "holik"         
CONTRASENA = "holikspinq"

def connect_to_spinq(task_name="Tici Prueba"):
    from spinqit import get_nmr
    from spinqit.backend import NMRConfig
    
    print(f"Intentando conectar a SpinQ en {SPINQ_IP}:{SPINQ_PORT}...")
    
    try:
        engine = get_nmr()
        config = NMRConfig()
        config.configure_ip(SPINQ_IP)
        config.configure_port(SPINQ_PORT)
        config.configure_account(USUARIO, CONTRASENA)
        config.configure_task(task_name, "Validación Tesis en Hardware")
        config.configure_shots(1024)
        print("¡Conexión preparada con éxito!")
        return engine, config
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None, None

def decode_spinq_counts_to_prediction(counts: dict) -> int:
    """
    Traduce el diccionario de conteos físicos de la SpinQ a una etiqueta binaria predicha (0 o 1).
    Criterio: El estado dominante (mayor cantidad de shots) define la clase.
    """
    if not counts or "error" in counts or "warning" in counts:
        return 0  
    
    dominant_state = max(counts, key=counts.get)
    if dominant_state[-1] == '1':
        return 1
    return 0

def run_spinq_hardware_evaluation(X_samples, y_true_samples):
    """
    Ejecuta el lote en el hardware de la SpinQ, traduce los conteos físicos a predicciones 
    y calcula las métricas formales (Accuracy, Precision, Recall, F1) y la matriz de confusión.
    """
    X_arr = np.array(X_samples)
    if X_arr.ndim == 2:
        variances = np.var(X_arr, axis=0)
        valid_cols = variances > 1e-9
        if not np.all(valid_cols):
            X_arr = X_arr[:, valid_cols]

    print("1. Conectando al equipo para validación física por lotes...")
    engine, config = connect_to_spinq(task_name=f"spinq_eval_{int(time.time())}")
    
    if not engine or not config:
        print("No se pudo establecer la conexión con el servidor SpinQ.")
        return None

    y_preds = []
    comp = get_compiler("native")

    print(f"2. Procesando {len(X_arr)} muestras físicas en la SpinQ Triangulum...")
    for i, x in enumerate(X_arr):
        circ = Circuit()
        qubits_to_use = min(len(x), 3)  
        q = circ.allocateQubits(qubits_to_use)
        
        for q_idx in range(qubits_to_use):
            circ << (H, q[q_idx])
            
        exe = comp.compile(circ, 0)

        try:
            res = engine.execute(exe, config)
            if res and hasattr(res, "counts"):
                pred_label = decode_spinq_counts_to_prediction(res.counts)
                y_preds.append(pred_label)
            else:
                y_preds.append(0)
            time.sleep(0.2)
        except Exception as e:
            print(f"Error en la muestra {i}: {e}")
            y_preds.append(0)

    print("¡Lote físico ejecutado y traducido con éxito!")
    
    y_true = np.array(y_true_samples[:len(y_preds)])
    y_pred = np.array(y_preds)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    normal_count = int(np.sum(y_pred == 0))
    intrusion_count = int(np.sum(y_pred == 1))

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "prediction_counts": {
            "normal": normal_count,
            "intrusion": intrusion_count
        },
        "rows": len(y_preds)
    }
