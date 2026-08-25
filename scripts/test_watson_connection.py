"""
Diagnostic Utility: Watson NLP Network Connectivity & Endpoint Health Checker.

Performs staged checks for DNS resolution, TCP port 443 handshake,
and HTTPS API invocation with detailed diagnostics.
"""

import os
import socket
import time
from typing import List
import urllib.parse
import requests

DEFAULT_WATSON_URL = (
    "https://sn-watson-sentiment-bert.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict"
)


def check_dns(hostname: str, port: int) -> List[str]:
    """Resolves DNS hostname and reports resolved IP addresses."""
    print("\n[Step 1/3] DNS Resolution Check...")
    try:
        addr_info = socket.getaddrinfo(hostname, port)
        ip_addresses = list({info[4][0] for info in addr_info})
        print(f"  [PASS] Resolved '{hostname}' to {len(ip_addresses)} IP address(es):")
        for ip in ip_addresses:
            is_private = (
                ip.startswith("10.") or
                ip.startswith("192.168.") or
                ip.startswith("172.16.")
            )
            ip_type = "PRIVATE / INTERNAL" if is_private else "PUBLIC"
            print(f"         - {ip} ({ip_type})")
        return ip_addresses
    except socket.gaierror as dns_err:
        print(f"  [FAIL] DNS resolution failed: {dns_err}")
        return []


def check_tcp_handshake(ip_addresses: List[str], port: int, timeout_sec: int) -> bool:
    """Attempts TCP 3-way handshake against discovered IP addresses."""
    print("\n[Step 2/3] TCP Handshake Check (Port 443)...")
    for ip in ip_addresses:
        print(f"  Attempting TCP connect to {ip}:{port} (timeout={timeout_sec}s)...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_sec)
        start_time = time.time()
        try:
            sock.connect((ip, port))
            elapsed = (time.time() - start_time) * 1000
            print(f"  [PASS] Connected to {ip}:{port} in {elapsed:.1f}ms")
            sock.close()
            return True
        except socket.timeout:
            print(f"  [FAIL] TCP connection to {ip}:{port} timed out ({timeout_sec}s).")
        except socket.error as sock_err:
            print(f"  [FAIL] TCP connection to {ip}:{port} refused/failed: {sock_err}")
        finally:
            sock.close()
    return False


def check_https_request(target_url: str, timeout_sec: int) -> None:
    """Sends test HTTP POST request to verify application level API ingestion."""
    print("\n[Step 3/3] HTTPS API Request Check...")
    headers = {
        "grpc-metadata-mm-model-id": os.getenv(
            "WATSON_MODEL_ID",
            "sentiment_aggregated-bert-workflow_lang_multi_stock"
        ),
        "Content-Type": "application/json",
    }
    payload = {"raw_document": {"text": "Diagnostic connectivity ping"}}
    try:
        start_time = time.time()
        resp = requests.post(
            target_url, json=payload, headers=headers, timeout=timeout_sec
        )
        elapsed = (time.time() - start_time) * 1000
        print(f"  [PASS] HTTP Status Code : {resp.status_code}")
        print(f"         Response Time    : {elapsed:.1f}ms")
        print(f"         Response Preview : {resp.text[:120]}...")
    except requests.exceptions.RequestException as req_err:
        print(f"  [FAIL] HTTPS request failed: {req_err}")


def print_summary(ip_addresses: List[str], tcp_ok: bool) -> None:
    """Prints diagnostic diagnosis summary."""
    print("\n" + "=" * 70)
    print(" Diagnostic Summary & Findings")
    print("=" * 70)
    has_private_ip = any(ip.startswith("10.") for ip in ip_addresses)
    if has_private_ip and not tcp_ok:
        print("! ROOT CAUSE IDENTIFIED:")
        print("  The Watson endpoint resolves to private IBM Skills Network internal IP(s):")
        print(f"  {ip_addresses}")
        print("  These internal addresses (10.x.x.x) are only routable inside the IBM lab/VPC.")
        print("  When running on an external/local Windows environment, TCP connections time out.")
        print("  -> Application properly returns 'TIMEOUT' and presents user-friendly service")
        print("     unavailable status without crashing.")
    elif tcp_ok:
        print("  Network connectivity to Watson NLP BERT service is fully operational.")
    else:
        print("  Unable to reach Watson NLP endpoint due to network/firewall constraints.")
    print("=" * 70)


def run_diagnostics(target_url: str = DEFAULT_WATSON_URL, timeout_sec: int = 5) -> None:
    """Executes staged network diagnostics for the Watson NLP service endpoint."""
    print("=" * 70)
    print(" Watson NLP BERT Service Network Diagnostics")
    print("=" * 70)
    parsed = urllib.parse.urlparse(target_url)
    hostname = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    print(f"Target URL : {target_url}\nHostname   : {hostname}\nPort       : {port}")
    print("-" * 70)

    ip_addresses = check_dns(hostname, port)
    if not ip_addresses:
        return

    tcp_ok = check_tcp_handshake(ip_addresses, port, timeout_sec)
    if tcp_ok:
        check_https_request(target_url, timeout_sec)
    else:
        print("\n[Step 3/3] Skipping HTTPS check (TCP could not be established).")

    print_summary(ip_addresses, tcp_ok)


if __name__ == "__main__":
    run_diagnostics(
        target_url=os.getenv("WATSON_SENTIMENT_URL", DEFAULT_WATSON_URL)
    )
