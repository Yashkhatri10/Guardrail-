import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.router import route


def main():
    parser = argparse.ArgumentParser(description='Run the pipeline on a JSONL file')
    parser.add_argument('file_path', help='Path to the JSONL file')
    args = parser.parse_args()

    results = []
    with open(args.file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get('text') if isinstance(data, dict) else data
            result = route(text)
            entry = {"line_number": line_number, "result": result}
            results.append(entry)
            print(f"{line_number}: {json.dumps(result)}")

    output_path = ROOT / "results" / "final_holdout_scores.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"source_file": args.file_path, "results": results}, f, indent=2)

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == '__main__':
    main()