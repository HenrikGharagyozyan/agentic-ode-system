import json
import sys

def print_structure(obj, indent=0, max_depth=3):
    if indent > max_depth:
        print(' ' * indent + '...')
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(' ' * indent + f'{k}:', end='')
            if isinstance(v, (dict, list)):
                print()
                print_structure(v, indent+2, max_depth)
            else:
                print(f' {v}')
    elif isinstance(obj, list):
        if len(obj) == 0:
            print(' ' * indent + '[]')
        else:
            print(' ' * indent + f'list of length {len(obj)}')
            # Show first few elements
            for i, item in enumerate(obj[:3]):
                print(' ' * (indent+2) + f'[{i}]:', end='')
                if isinstance(item, (dict, list)):
                    print()
                    print_structure(item, indent+4, max_depth)
                else:
                    print(f' {item}')
            if len(obj) > 3:
                print(' ' * (indent+2) + '...')
    else:
        print(' ' * indent + str(obj))

def main():
    json_path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    print_structure(data)

if __name__ == '__main__':
    main()