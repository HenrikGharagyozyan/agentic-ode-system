import json
import sys

def main():
    # Path to transcript.json
    transcript_path = '/home/henrik/Projects/agentic-ode-system/runs/20261009-201921/transcript.json'
    try:
        with open(transcript_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {transcript_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract runs
    runs = data.get('runs', [])
    if len(runs) < 4:
        print("Error: Not enough runs in transcript", file=sys.stderr)
        sys.exit(1)

    run3 = runs[3]  # zero-indexed, so run 3 is the fourth run
    test_rmse = run3.get('test_rmse')
    train_rmse = run3.get('train_rmse')

    if test_rmse is None or train_rmse is None:
        print("Error: Missing RMSE values in run 3", file=sys.stderr)
        sys.exit(1)

    # Print progress log (optional)
    print(f"Loaded transcript from {transcript_path}")
    print(f"Found {len(runs)} runs")
    print(f"Using run 3 (index 3)")
    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    # Required output lines
    print(f"FREE_RUN_TRAIN_RMSE: {train_rmse}")
    print(f"FREE_RUN_TEST_RMSE: {test_rmse}")

if __name__ == '__main__':
    main()