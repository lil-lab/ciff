import random
import socket

MIN_PORT = 1024
MAX_PORT = 65535


def check_if_port_available(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    can_connect = True
    try:
        s.bind(("127.0.0.1", port))
    except socket.error:
        can_connect = False
    s.close()

    return can_connect


def find_k_ports(k, max_try=10):
    for _ in range(max_try):
        start_port = random.randint(MIN_PORT, MAX_PORT - k)
        port_list = range(start_port, start_port + k)
        all_available = True
        for port in port_list:
            if not check_if_port_available(port):
                all_available = False
                break
        if all_available:
            return port_list
    return None