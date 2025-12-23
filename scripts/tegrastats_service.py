#!/usr/bin/env python3
"""
Tegrastats Host Service
Runs on the Jetson host and writes tegrastats data to a JSON file
that can be read by Docker containers via volume mount.
"""

import subprocess
import re
import json
import time
import os
import signal
import sys
from datetime import datetime

OUTPUT_FILE = "/tmp/tegrastats.json"
INTERVAL_MS = 1000  # 1 second

running = True


def signal_handler(signum, frame):
    global running
    running = False
    print("Shutting down tegrastats service...")


def parse_tegrastats_line(line: str) -> dict:
    """Parse a single tegrastats output line."""
    result = {
        "cpu_temp": None,
        "gpu_temp": None,
        "gpu_power_mw": None,
        "total_power_mw": None,
        "cpu_power_mw": None,
        "available": False,
        "timestamp": datetime.utcnow().isoformat()
    }

    if not line:
        return result

    try:
        # CPU temperature: cpu@38.125C
        cpu_temp_match = re.search(r'cpu@([\d.]+)C', line)
        if cpu_temp_match:
            result["cpu_temp"] = float(cpu_temp_match.group(1))

        # GPU temperature: gpu@40.343C
        gpu_temp_match = re.search(r'gpu@([\d.]+)C', line)
        if gpu_temp_match:
            result["gpu_temp"] = float(gpu_temp_match.group(1))

        # GPU Power: VDD_GPU 9488mW or VDD_GPU_SOC 9488mW
        gpu_power_match = re.search(r'VDD_GPU(?:_SOC)?\s+(\d+)mW', line)
        if gpu_power_match:
            result["gpu_power_mw"] = int(gpu_power_match.group(1))

        # CPU Power: VDD_CPU_SOC_MSS 8697mW or VDD_CPU 8697mW
        cpu_power_match = re.search(r'VDD_CPU(?:_SOC_MSS|_CV)?\s+(\d+)mW', line)
        if cpu_power_match:
            result["cpu_power_mw"] = int(cpu_power_match.group(1))

        # Total Power: VIN 36834mW
        total_power_match = re.search(r'VIN\s+(\d+)mW', line)
        if total_power_match:
            result["total_power_mw"] = int(total_power_match.group(1))

        # Mark as available if we got at least one value
        if any([result["cpu_temp"], result["gpu_temp"], result["gpu_power_mw"], result["total_power_mw"]]):
            result["available"] = True

    except Exception as e:
        print(f"Parse error: {e}")

    return result


def write_data(data: dict):
    """Write data to JSON file atomically."""
    temp_file = OUTPUT_FILE + ".tmp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        os.rename(temp_file, OUTPUT_FILE)
    except Exception as e:
        print(f"Write error: {e}")


def main():
    global running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"Starting tegrastats service, writing to {OUTPUT_FILE}")

    try:
        # Start tegrastats process
        proc = subprocess.Popen(
            ['tegrastats', '--interval', str(INTERVAL_MS)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        while running:
            line = proc.stdout.readline()
            if not line:
                break

            data = parse_tegrastats_line(line.strip())
            write_data(data)

    except FileNotFoundError:
        print("tegrastats not found - not running on Jetson?")
        # Write empty data
        write_data({
            "available": False,
            "error": "tegrastats not found",
            "timestamp": datetime.utcnow().isoformat()
        })
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'proc' in locals():
            proc.terminate()
            proc.wait()

    print("Tegrastats service stopped")


if __name__ == "__main__":
    main()
