import json
import sys

def find_validation_rmse(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            # Check if key looks like validation rmse
            if isinstance(v, (int, float)) and 'validation' in k.lower() and 'rmse' in k.lower():
                return v
            # Recurse
            result = find_validation_rmse(v, new_path)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            result = find_validation_rmse(item, f"{path}[{i}]")
            if result is not None:
                return result
    return None

def main():
    json_path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    rmse = find_validation_rmse(data)
    if rmse is None:
        # Fallback: try to find any numeric value under a key containing 'validation'
        def find_any_validation(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if 'validation' in k.lower():
                        if isinstance(v, (int, float)):
                            return v
                        if isinstance(v, dict):
                            for k2, v2 in v.items():
                                if 'rmse' in k2.lower():
                                    return v2
                        result = find_any_validation(v)
                        if result is not None:
                            return result
            elif isinstance(obj, list):
                for item in obj:
                    result = find_any_validation(item)
                    if result is not None:
                        return result
            return None
        rmse = find_any_validation(data)
    
    if rmse is None:
        print("Validation RMSE not found in transcript.json", file=sys.stderr)
        sys.exit(1)
    
    print(rmse)

if __name__ == '__main__':
    main()