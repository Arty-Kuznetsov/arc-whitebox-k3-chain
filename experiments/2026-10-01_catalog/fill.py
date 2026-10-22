"""Refresh the generated blocks of RESULTS.md: <!-- T1 -->, <!-- GRID3 -->, <!-- STACK --> from report.py."""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
res = HERE / "RESULTS.md"
s = res.read_text()
for k, arg in (("T1", "t1"), ("GRID3", "grid3"), ("STACK", "stack")):
    out = subprocess.run([sys.executable, "report.py", arg], capture_output=True, text=True, cwd=HERE).stdout.strip()
    s = re.sub(rf"<!-- {k} -->.*?<!-- /{k} -->", lambda m: f"<!-- {k} -->\n{out}\n<!-- /{k} -->", s, flags=re.S)
res.write_text(s)
print("filled")
