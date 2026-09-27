import subprocess
import sys

for i in range(1,5):
    subprocess.run([sys.executable, "netmonitor_analysis.py", "--clf", f"{i}.clf", "--kml", f"{i}.kml"], check=True)