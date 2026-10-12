import json
import os

transcript_path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
with open(transcript_path, 'r') as f:
    data = json.load(f)

print('Keys:', list(data.keys()))
print('Model:', data.get('model'))
print('Task file:', data.get('task_file'))
print('Run dir:', data.get('run_dir'))
print('Started:', data.get('started'))
print('Finished:', data.get('finished'))
print('Stop reason:', data.get('stop_reason'))
print('Number of decisions:', len(data.get('decisions', [])))
print('Number of dialogue entries:', len(data.get('dialogue', [])))
print('Number of runs:', len(data.get('runs', [])))

for i, run in enumerate(data.get('runs', [])):
    print(f'\nRun {i}:')
    for key, value in run.items():
        if key not in ['stdout', 'stderr']:  # avoid large fields
            print(f'  {key}: {value}')
        else:
            print(f'  {key}: <length {len(value) if value else 0}>')
    # If there is stdout, maybe we can see if it contains RMSE lines
    stdout = run.get('stdout', '')
    if stdout:
        lines = stdout.split('\n')
        for line in lines:
            if 'FREE_RUN_TRAIN_RMSE' in line or 'FREE_RUN_TEST_RMSE' in line or 'test_rmse' in line or 'train_rmse' in line:
                print(f'    {line.strip()}')