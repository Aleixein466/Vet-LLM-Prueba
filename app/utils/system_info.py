"""Utilidades de sistema (CPU/RAM/disco) para la página de estado."""
from __future__ import annotations
import os
import shutil
import torch

def info() -> dict:
    try:
        import psutil
        ram = psutil.virtual_memory()
        ram_txt = f"{ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB ({ram.percent}%)"
        cpu_pct = psutil.cpu_percent(interval=0.5)
    except ImportError:
        ram_txt, cpu_pct = "NO DISPONIBLE", "NO DISPONIBLE"
    d = shutil.disk_usage("D:/")
    return {
        "cpu_logicos": os.cpu_count(),
        "cpu_uso": cpu_pct,
        "ram": ram_txt,
        "disco_D_libre_GB": round(d.free / 1e9, 1),
        "torch": torch.__version__,
        "dispositivo": "CPU",
        "threads_torch": torch.get_num_threads(),
    }
