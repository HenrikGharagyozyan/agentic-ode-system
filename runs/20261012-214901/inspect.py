import json
import sys

path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
try:
    with open(path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)

# If data is a dict, print keys
if isinstance(data, dict):
    print("Keys:", list(data.keys()))
    # Optionally print a few items
    for k, v in list(data.items())[:5]:
        print(f"{k}: {v}")
elif isinstance(data, list):
    print(f"List length: {len(data)}")
    if len(data) > 0:
        print("First element:", data[0])
        if isinstance(data[0], dict):
            print("First element keys:", list(data[0].keys()))
else:
    print("Type:", type(data))
    print("Value:", data)