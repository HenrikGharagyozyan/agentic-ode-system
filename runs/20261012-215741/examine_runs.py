import json
import os

run_dir = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921'
transcript_path = os.path.join(run_dir, 'transcript.json')
with open(transcript_path, 'r') as f:
    data = json.load(f)

print('Keys:', list(data.keys()))
print('Number of runs:', len(data.get('runs', [])))

for i, run in enumerate(data.get('runs', [])):
    print(f'\\nRun {i}:')
    if isinstance(run, dict):
        for k, v in run.items():
            if isinstance(v, (int, float)):
                print(f'  {k}: {v}')
            elif isinstance(v, str) and len(v) < 100:
                print(f'  {k}: {v}')
            else:
                print(f'  {k}: {type(v)}')
                if isinstance(v, dict):
                    # print subkeys
                    for sk, sv in v.items():
                        if isinstance(sv, (int, float)):
                            print(f'    {sk}: {sv}')
                        elif isinstance(v, dict) and len(str(sv)) < 100:
                            print(f'    {sk}: {sv}')
    else:
        print(f'  {run}')