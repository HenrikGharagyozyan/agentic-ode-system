import os
import json

run_dir = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921'
print('Files in run_dir:')
for f in sorted(os.listdir(run_dir)):
    print(f)

# Let's examine transcript.json more deeply
transcript_path = os.path.join(run_dir, 'transcript.json')
print('\\nLoading transcript.json...')
with open(transcript_path, 'r') as f:
    data = json.load(f)

# Let's see the keys
print('\\nTop-level keys:', list(data.keys()))

# Let's recursively search for any numeric values with keys containing 'validation' or 'rmse'
def search(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f'{path}.{k}' if path else k
            if isinstance(v, (int, float)):
                if 'validation' in k.lower() and 'rmse' in k.lower():
                    print(f'Found: {new_path} = {v}')
                elif 'validation' in k.lower():
                    print(f'Validation-related: {new_path} = {v}')
                elif 'rmse' in k.lower():
                    print(f'RMSE-related: {new_path} = {v}')
            else:
                search(v, new_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            search(item, f'{path}[{i}]')

print('\\nSearch results:')
search(data)