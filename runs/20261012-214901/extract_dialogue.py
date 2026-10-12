import json
import sys

path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
try:
    with open(path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)

dialogue = data.get('dialogue', [])
if not dialogue:
    print("No dialogue found")
    sys.exit(0)

print(f"Number of dialogue entries: {len(dialogue)}")
for i, entry in enumerate(dialogue):
    print(f"\nEntry {i}:")
    if isinstance(entry, dict):
        for k, v in entry.items():
            if k in ['content', 'text', 'message']:
                print(f"  {k}: {v[:200]}..." if isinstance(v, str) and len(v) > 200 else f"  {k}: {v}")
            elif 'rmse' in k.lower() or 'error' in k.lower():
                print(f"  {k}: {v}")
    else:
        print(f"  {entry}")

# Also look for any printed lines in the runs' stdout
runs = data.get('runs', [])
for i, run in enumerate(runs):
    if isinstance(run, dict):
        stdout = run.get('stdout', '')
        if stdout:
            print(f"\nRun {i} stdout (first 500 chars):")
            print(stdout[:500])
            # Look for lines with RMSE
            for line in stdout.split('\\n'):
                if 'rmse' in line.lower() or 'error' in line.lower():
                    print(f"  RMSE line: {line}")