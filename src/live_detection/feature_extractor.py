from __future__ import annotations

from collections.abc import Iterable
from statistics import pstdev

from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.packet import Packet


def extract_live_features(packets: Iterable[Packet], duration_seconds: float) -> dict[str, float]:
    packet_list = list(packets)
    safe_duration = max(float(duration_seconds), 1e-9)

    total_packets = len(packet_list)
    tcp_packets = 0
    udp_packets = 0
    icmp_packets = 0
    syn_packets = 0
    ack_packets = 0
    rst_packets = 0
    fin_packets = 0
    psh_packets = 0
    tcp_option_packets = 0
    unique_src_ips: set[str] = set()
    unique_dst_ports: set[int] = set()
    src_ip_counts: dict[str, int] = {}
    dst_port_counts: dict[int, int] = {}
    packet_sizes: list[int] = []
    packet_timestamps: list[float] = []
    ttl_values: list[int] = []
    packets_with_payload = 0
    http_ports_packets = 0
    dns_port_packets = 0
    ntp_port_packets = 0
    snmp_port_packets = 0
    ssdp_port_packets = 0
    traceroute_port_packets = 0

    for packet in packet_list:
        try:
            packet_size = len(packet)
            packet_sizes.append(packet_size)
        except TypeError:
            continue

        packet_time = getattr(packet, "time", None)
        if packet_time is not None:
            try:
                packet_timestamps.append(float(packet_time))
            except (ValueError, TypeError):
                pass

        if IP in packet:
            src_ip = str(packet[IP].src)
            unique_src_ips.add(src_ip)
            src_ip_counts[src_ip] = src_ip_counts.get(src_ip, 0) + 1
            try:
                ttl_values.append(int(packet[IP].ttl))
            except (AttributeError, ValueError, TypeError):
                pass

        if TCP in packet:
            tcp_packets += 1
            dst_port = int(packet[TCP].dport)
            unique_dst_ports.add(dst_port)
            dst_port_counts[dst_port] = dst_port_counts.get(dst_port, 0) + 1
            flags = packet[TCP].flags
            if int(flags) & 0x02:
                syn_packets += 1
            if int(flags) & 0x10:
                ack_packets += 1
            if int(flags) & 0x04:
                rst_packets += 1
            if int(flags) & 0x01:
                fin_packets += 1
            if int(flags) & 0x08:
                psh_packets += 1
            if getattr(packet[TCP], "options", None):
                tcp_option_packets += 1
            if dst_port in {80, 443, 8080, 8443}:
                http_ports_packets += 1
            elif dst_port == 53:
                dns_port_packets += 1
            elif dst_port == 123:
                ntp_port_packets += 1
            elif dst_port == 161:
                snmp_port_packets += 1
            elif dst_port == 1900:
                ssdp_port_packets += 1
            elif dst_port == 33434:
                traceroute_port_packets += 1

        if UDP in packet:
            udp_packets += 1
            dst_port = int(packet[UDP].dport)
            unique_dst_ports.add(dst_port)
            dst_port_counts[dst_port] = dst_port_counts.get(dst_port, 0) + 1
            if dst_port in {80, 443, 8080, 8443}:
                http_ports_packets += 1
            elif dst_port == 53:
                dns_port_packets += 1
            elif dst_port == 123:
                ntp_port_packets += 1
            elif dst_port == 161:
                snmp_port_packets += 1
            elif dst_port == 1900:
                ssdp_port_packets += 1
            elif dst_port == 33434:
                traceroute_port_packets += 1

        if ICMP in packet:
            icmp_packets += 1

        try:
            payload_size = len(bytes(packet.payload))
            if payload_size > 0:
                packets_with_payload += 1
        except (TypeError, AttributeError):
            pass

    avg_packet_size = sum(packet_sizes) / total_packets if total_packets else 0.0
    min_packet_size = min(packet_sizes) if packet_sizes else 0
    max_packet_size = max(packet_sizes) if packet_sizes else 0
    std_packet_size = pstdev(packet_sizes) if len(packet_sizes) > 1 else 0.0
    if len(packet_timestamps) > 1:
        packet_timestamps.sort()
        inter_arrival_times = [
            current - previous for previous, current in zip(packet_timestamps, packet_timestamps[1:]) if current >= previous
        ]
    else:
        inter_arrival_times = []
    avg_inter_arrival = sum(inter_arrival_times) / len(inter_arrival_times) if inter_arrival_times else 0.0
    std_inter_arrival = pstdev(inter_arrival_times) if len(inter_arrival_times) > 1 else 0.0
    top_src_ip_share = max(src_ip_counts.values()) / total_packets if total_packets and src_ip_counts else 0.0
    top_dst_port_share = max(dst_port_counts.values()) / total_packets if total_packets and dst_port_counts else 0.0
    avg_ttl = sum(ttl_values) / len(ttl_values) if ttl_values else 0.0
    std_ttl = pstdev(ttl_values) if len(ttl_values) > 1 else 0.0

    return {
        "duration_seconds": round(float(duration_seconds), 4),
        "total_packets": total_packets,
        "tcp_packets": tcp_packets,
        "udp_packets": udp_packets,
        "icmp_packets": icmp_packets,
        "syn_packets": syn_packets,
        "ack_packets": ack_packets,
        "rst_packets": rst_packets,
        "fin_packets": fin_packets,
        "psh_packets": psh_packets,
        "tcp_option_packets": tcp_option_packets,
        "unique_src_ips": len(unique_src_ips),
        "unique_dst_ports": len(unique_dst_ports),
        "packet_rate": round(total_packets / safe_duration, 4),
        "avg_packet_size": round(avg_packet_size, 4),
        "min_packet_size": min_packet_size,
        "max_packet_size": max_packet_size,
        "std_packet_size": round(std_packet_size, 4),
        "avg_inter_arrival": round(avg_inter_arrival, 6),
        "std_inter_arrival": round(std_inter_arrival, 6),
        "top_src_ip_share": round(top_src_ip_share, 4),
        "top_dst_port_share": round(top_dst_port_share, 4),
        "packets_with_payload": packets_with_payload,
        "http_ports_packets": http_ports_packets,
        "dns_port_packets": dns_port_packets,
        "ntp_port_packets": ntp_port_packets,
        "snmp_port_packets": snmp_port_packets,
        "ssdp_port_packets": ssdp_port_packets,
        "traceroute_port_packets": traceroute_port_packets,
        "avg_ttl": round(avg_ttl, 4),
        "std_ttl": round(std_ttl, 4),
    }
