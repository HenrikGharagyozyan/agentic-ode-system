import json
import re

path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
with open(path, 'r') as f:
    data = json.load(f)

runs = data.get('runs', [])
if len(runs) <= 3:
    print("Not enough runs")
    exit(0)

run3 = runs[3]
if not isinstance(run3, dict):
    print("Run 3 is not a dict")
    exit(0)

stdout = run3.get('stdout', '')
stderr = run3.get('stderr', '')
exit_code = run3.get('exit_code', None)

print(f"Exit code: {exit_code}")
print(f"Stderr length: {len(stderr)}")
if stderr:
    print("Stderr (first 500 chars):")
    print(stderr[:500])
print("Stdout:")
print(stdout)
print("--- End of stdout ---")

# Look for free-run lines
pattern_train = re.compile(r'FREE_RUN_TRAIN_RMSE:\s*([0-9]+\\.?[0-9]*)')
pattern_test = re.compile(r'FREE_RUN_TEST_RMSE:\s*([0-9]+\\.?[0-9]*)')

train_match = pattern_train.search(stdout)
test_match = pattern_test.search(stdout)

if train_match:
    print(f"Found FREE_RUN_TRAIN_RMSE: {train_match.group(1)}")
else:
    print("FREE_RUN_TRAIN_RMSE not found in stdout")
if test_match:
    print(f"Found FREE_RUN_TEST_RMSE: {test_match.group(1)}")
else:
    print("FREE_RUN_TEST_RMSE not found in stdout")

# Also, check if the run dict has test_rmse and train_rmse
print(f"Run dict test_rmse: {run3.get('test_rmse')}")
print(f"Run dict train_rmse: {run3.get('train_rmse')}")