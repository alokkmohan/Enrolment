import subprocess
import os

scripts = [
    "scripts/data_loader.py",
    "scripts/batch1_indices.py",
    "scripts/batch2_indices.py",
    "scripts/batch3_indices.py",
    "scripts/batch4_indices.py",
    "scripts/batch5_master.py",
]

for script in scripts:
    if os.path.exists(script):
        print(f"\n{'='*50}")
        print(f"Running: {script}")
        print('='*50)
        subprocess.run(["python", script], check=True)
    else:
        print(f"SKIPPING (not yet created): {script}")
