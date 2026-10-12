import json
import sys
import re

path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
try:
    with open(path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)

runs = data.get('runs', [])
if not runs:
    print("No runs found")
    sys.exit(0)

pattern_train = re.compile(r'FREE_RUN_TRAIN_RMSE:\s*([0-9]+\\.?[0-9]*)')
pattern_test = re.compile(r'FREE_RUN_TEST_RMSE:\s*([0-9]+\\.?[0-9]*)')

for i, run in enumerate(runs):
    if not isinstance(run, dict):
        continue
    stdout = run.get('stdout', '')
    stderr = run.get('stderr', '')
    exit_code = run.get('exit_code', None)
    print(f"\nRun {i}: exit_code={exit_code}")
    # Search for the patterns
    train_match = pattern_train.search(stdout)
    test_match = pattern_test.search(stdout)
    if train_match:
        print(f"  FREE_RUN_TRAIN_RMSE: {train_match.group(1)}")
    if test_match:
        print(f"  FREE_RUN_TEST_RMSE: {test_match.group(1)}")
    # If not found, maybe print a few lines of stdout for context
    if not (train_match or test_match):
        # Show first 10 lines of stdout
        lines = stdout.split('\\n')
        for j, line in enumerate(lines[:10]):
            if line.strip():
                print(f"    {line}")
        # Also look for any RMSE lines
        for line in lines:
            if 'rmse' in line.lower():
                print(f"    RMSE line: {line}")

# If we found values, we can also compute or just report.
# We'll also check the dialogue for any mention of RMSE.
dialogue = data.get('dialogue', [])
for i, entry in enumerate(dialogue):
    if isinstance(entry, dict):
        content = entry.get('content', '') or entry.get('text', '') or entry.get('message', '')
        if isinstance(content, str):
            if 'free_run' in content.lower() and 'rmse' in content.lower():
                print(f"\nDialogue entry {i} mentions free-run RMSE: {content[:200]}")