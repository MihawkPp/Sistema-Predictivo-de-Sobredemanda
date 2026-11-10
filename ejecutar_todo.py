"""Ejecuta el prototipo completo: preparación, exploración y modelado."""
import subprocess
import sys
from pathlib import Path

CODIGO = Path(__file__).resolve().parent / "codigo"
for paso in ["01_preparar_datos.py", "02_explorar.py", "03_modelar.py"]:
    print(f"\n=== {paso} ===", flush=True)
    subprocess.run([sys.executable, str(CODIGO / paso)], cwd=CODIGO, check=True)
print("\nListo. Revise la carpeta 'resultados'.")
