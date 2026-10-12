import json
import sys

path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
try:
    with open(path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)

# The runs are under the key 'runs'
if 'runs' not in data:
    print("Key 'runs' not found in data")
    sys.exit(1)

runs = data['runs']
if not isinstance(runs, list):
    print("'runs' is not a list")
    sys.exit(1)

print(f"Number of runs: {len(runs)}")
for i, run in enumerate(runs):
    print(f"\nRun {i}:")
    if isinstance(run, dict):
        # Print all keys and values, but we are looking for RMSE
        for k, v in run.items():
            if 'rmse' in k.lower() or 'error' in k.lower():
                print(f"  {k}: {v}")
        # If we didn't find, print a few keys for context
        if not any('rmse' in k.lower() or 'error' in k.lower() for k in run.keys()):
            print("  (No RMSE/error keys found, showing first 5 items):")
            for k, v in list(run.items())[:5]:
                print(f"    {k}: {v}")
    else:
        print(f"  Non-dict item: {run}")

# Now, let's try to find the free-run RMSE specifically.
# We'll look for a key that contains 'free' and 'rmse' or similar.
free_run_rmse = None
for run in runs:
    if isinstance(run, dict):
        for k, v in run.items():
            if 'free' in k.lower() and 'rmse' in k.lower():
                free_run_rmse = v
                break
        if free_run_rmse is not None:
            break

if free_run_rmse is not None:
    print(f"\nFound free-run RMSE: {free_run_rmse}")
else:
    print("\nCould not find free-run RMSE with automatic search.")
    # Let's try to print the entire runs structure for manual inspection (but limit output)
    print("\nDetailed runs (first run only):")
    if len(runs) > 0:
        first = runs[0]
        if isinstance(first, dict):
            for k, v in first.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {first}")