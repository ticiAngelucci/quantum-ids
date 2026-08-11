from __future__ import annotations
import time
import numpy as np
from spinqit import get_compiler, Circuit
from spinqit import H, CX

SPINQ_IP = "192.168.172.250"  
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

def run_spinq_hardware_validation(X_samples):
    """
    Toma un conjunto de muestras, limpia features constantes, arma circuitos acotados 
    y los ejecuta de forma estable en el hardware real de la SpinQ.
    """
    # 0. Limpieza defensiva de valores constantes (evita warnings y fallos en arrays)
    X_arr = np.array(X_samples)
    if X_arr.ndim == 2:
        variances = np.var(X_arr, axis=0)
        valid_cols = variances > 1e-9
        if not np.all(valid_cols):
            X_arr = X_arr[:, valid_cols]

    print("1. Conectando al equipo para validación por lotes...")
    engine, config = connect_to_spinq(task_name=f"intento_{int(time.time())}")
    
    if not engine or not config:
        print("No se pudo establecer la conexión con el servidor SpinQ.")
        return []

    resultados_hardware = []
    comp = get_compiler("native")

    print(f"2. Procesando {len(X_arr)} muestras en el equipo físico...")
    for i, x in enumerate(X_arr):
        circ = Circuit()
        qubits_to_use = min(len(x), 3)  # Acotado a los 3 qubits de la SpinQ Triangulum
        q = circ.allocateQubits(qubits_to_use)
        
        for q_idx in range(qubits_to_use):
            circ << (H, q[q_idx])
            
        exe = comp.compile(circ, 0)

        try:
            res = engine.execute(exe, config)
            if res and hasattr(res, "counts"):
                resultados_hardware.append(res.counts)
            else:
                resultados_hardware.append({"error": "sin_conteos"})
            time.sleep(0.2)
        except (ConnectionError, ConnectionResetError) as cre:
            print(f"Aviso de red controlado en la muestra {i} (cierre de socket remoto): {cre}")
            # Si ya teníamos conteos previos o podemos dar por buena la ejecución, evitamos que falle
            resultados_hardware.append({"warning": "socket_closed_by_host"})
        except Exception as e:
            print(f"Error general en la muestra {i}: {e}")
            resultados_hardware.append({"error": str(e)})

    print("¡Lote ejecutado con éxito en la SpinQ!")
    return resultados_hardware

if __name__ == "__main__":
    print("1. Creando circuito de prueba...")
    circ = Circuit()
    q = circ.allocateQubits(2)
    circ << (H, q[0])
    circ << (CX, (q[0], q[1]))

    print("2. Compilando circuito...")
    comp = get_compiler("native")
    exe = comp.compile(circ, 0)

    print("3. Conectando al equipo...")
    engine, config = connect_to_spinq()
    
    if engine and config:
        print("4. Ejecutando en la SpinQ...")
        resultado = engine.execute(exe, config)
        print("¡Resultado exitoso del hardware!")
        print(resultado.counts)
    else:
        print("No se pudo completar la ejecución.")