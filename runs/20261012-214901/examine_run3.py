import json
import sys

path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
try:
    with open(path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)

runs = data.get('runs', [])
if len(runs) <= 3:
    print("Not enough runs")
    sys.exit(0)

run3 = runs[3]
print("Run 3 keys:", list(run3.keys()) if isinstance(run3, dict) else "Not a dict")
if isinstance(run3, dict):
    for k, v in run3.items():
        print(f"  {k}: {v}")
        # If the value is large, we might want to truncate
        if isinstance(v, str) and len(v) > 500:
            print(f"    (length {len(v)})")
            # print first 200 and last 200
            print(f"    First 200: {v[:200]}")
            print(f"    Last 200: {v[-200:]}")