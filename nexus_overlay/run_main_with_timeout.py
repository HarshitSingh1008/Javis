"""
WRAPPER — runs `main.py` with a hard timeout and captures ALL output to files.
Usage:  python nexus_overlay/run_main_with_timeout.py [--seconds N] [--args ...]
"""
import sys, subprocess, time, pathlib

PY = pathlib.Path(sys.executable)
ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"
OUT = ROOT / "nexus_overlay" / "_main_capture_out.log"
ERR = ROOT / "nexus_overlay" / "_main_capture_err.log"

seconds = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 15
args = [str(PY), str(MAIN)] + sys.argv[2:] if len(sys.argv) > 2 else [str(PY), str(MAIN), "--verbose"]
print(f"Running: {' '.join(args)}")
print(f"Timeout: {seconds}s")
print(f"Capturing stdout -> {OUT}  stderr -> {ERR}")
print("-" * 70)

proc = subprocess.Popen(
    args,
    cwd=str(ROOT),
    stdout=open(OUT, "w", encoding="utf-8"),
    stderr=open(ERR, "w", encoding="utf-8"),
    text=True,
)

try:
    proc.wait(timeout=seconds)
    rc = proc.returncode
    print(f"main.py exited after {seconds}s with code {rc}")
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
    rc = -999
    print(f"main.py KILLED after {seconds}s (still running — likely blocked on overlay wait)")

print("-" * 70)
print("=== STDOUT ===")
try:
    print(OUT.read_text(encoding="utf-8")[-4000:] if OUT.exists() else "(no stdout file)")
except Exception as e:
    print("stdout read error:", e)
print("=== STDERR ===")
try:
    print(ERR.read_text(encoding="utf-8")[-2000:] if ERR.exists() else "(no stderr file)")
except Exception as e:
    print("stderr read error:", e)

sys.exit(rc)
