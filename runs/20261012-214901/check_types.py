import json
transcript_path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
with open(transcript_path, 'r') as f:
    data = json.load(f)

for key, value in data.items():
    print(f'{key}: {type(value)}')
    if isinstance(value, list):
        print(f'  length: {len(value)}')
        if len(value) > 0:
            print(f'  first element type: {type(value[0])}')
    elif isinstance(value, dict):
        print(f'  keys: {list(value.keys())}')
    else:
        print(f'  value: {value}')