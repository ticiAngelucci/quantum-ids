from __future__ import annotations

from collections.abc import Iterable

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
    unique_src_ips: set[str] = set()
    unique_dst_ports: set[int] = set()
    total_packet_size = 0

    for packet in packet_list:
        try:
            total_packet_size += len(packet)
        except Exception:
            continue

        if IP in packet:
            unique_src_ips.add(str(packet[IP].src))

        if TCP in packet:
            tcp_packets += 1
            unique_dst_ports.add(int(packet[TCP].dport))
            flags = packet[TCP].flags
            if int(flags) & 0x02:
                syn_packets += 1

        if UDP in packet:
            udp_packets += 1
            unique_dst_ports.add(int(packet[UDP].dport))

        if ICMP in packet:
            icmp_packets += 1

    avg_packet_size = total_packet_size / total_packets if total_packets else 0.0

    return {
        "duration_seconds": round(float(duration_seconds), 4),
        "total_packets": total_packets,
        "tcp_packets": tcp_packets,
        "udp_packets": udp_packets,
        "icmp_packets": icmp_packets,
        "syn_packets": syn_packets,
        "unique_src_ips": len(unique_src_ips),
        "unique_dst_ports": len(unique_dst_ports),
        "packet_rate": round(total_packets / safe_duration, 4),
        "avg_packet_size": round(avg_packet_size, 4),
    }
