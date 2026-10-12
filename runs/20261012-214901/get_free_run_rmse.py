import json
import re

path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
with open(path, 'r') as f:
    data = json.load(f)

runs = data.get('runs', [])
if not runs:
    print("No runs")
    exit(0)

# Patterns to look for
pattern_train = re.compile(r'FREE_RUN_TRAIN_RMSE:\s*([0-9]+\\.?[0-9]*)')
pattern_test = re.compile(r'FREE_RUN_TEST_RMSE:\s*([0-9]+\\.?[0-9]*)')

found_train = None
found_test = None

for i, run in enumerate(runs):
    if not isinstance(run, dict):
        continue
    stdout = run.get('stdout', '')
    # Search
    m_train = pattern_train.search(stdout)
    m_test = pattern_test.search(stdout)
    if m_train:
        found_train = float(m_train.group(1))
    if m_test:
        found_test = float(m_test.group(1))
    # If we found both, we can break but we'll continue to see if there are multiple
    if found_train is not None and found_test is not None:
        break

# If not found in runs, try dialogue
if found_train is None or found_test is None:
    dialogue = data.get('dialogue', [])
    for entry in dialogue:
        if isinstance(entry, dict):
            content = entry.get('content', '') or entry.get('text', '') or entry.get('message', '')
            if isinstance(content, str):
                if found_train is None:
                    m_train = pattern_train.search(content)
                    if m_train:
                        found_train = float(m_train.group(1))
                if found_test is None:
                    m_test = pattern_test.search(content)
                    if m_test:
                        found_test = float(m_test.group(1))
                if found_train is not None and found_test is not None:
                    break

print(f"FREE_RUN_TRAIN_RMSE: {found_train if found_train is not None else 'Not found'}")
print(f"FREE_RUN_TEST_RMSE: {found_test if found_test is not None else 'Not found'}")

# Also, if we have the test_rmse and train_rmse from the run dict, we can print them as fallback
for i, run in enumerate(runs):
    if isinstance(run, dict):
        if 'test_rmse' in run and run['test_rmse'] is not None:
            print(f"Run {i} test_rmse: {run['test_rmse']}")
        if 'train_rmse' in run and run['train_rmse'] is not None:
            print(f"Run {i} train_rmse: {run['train_rmse']}")