#!/usr/bin/env python3

'''

chmod +x split_grasp_results.py

./split_grasp_results.py \
  --input output/Qwen3-4B-Instruct-2507/all_results.json

'''



import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


ERROR_PATTERNS = [
    "error executing sparql query",
    "sparql execution failed",
    "call to function execute returned an error",
    "failed to parse input",
    "parse error",
    "traceback",
    "exception",
    "no output",
]


def safe_slug(text: str, max_len: int = 90) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "question"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_output(record: dict[str, Any]) -> dict[str, Any] | None:
    parsed = record.get("parsed_output")
    if not isinstance(parsed, dict):
        return None

    output = parsed.get("output")
    if not isinstance(output, dict):
        return None

    return output


def result_has_error(result_text: Any) -> bool:
    if not isinstance(result_text, str):
        return True

    normalized = result_text.lower()

    return any(pattern in normalized for pattern in ERROR_PATTERNS)


def classify_record(record: dict[str, Any]) -> tuple[bool, str]:
    """
    Returns:
      (is_valid, reason)
    """
    if record.get("status") != "ok":
        return False, f"script_status:{record.get('status')}"

    output = get_output(record)
    if output is None:
        return False, "missing_or_null_output"

    if output.get("type") != "answer":
        return False, f"output_type:{output.get('type')}"

    sparql = output.get("sparql")
    if not isinstance(sparql, str) or not sparql.strip():
        return False, "missing_sparql"

    result = output.get("result")
    if not isinstance(result, str) or not result.strip():
        return False, "missing_execution_result"

    if result_has_error(result):
        return False, "execution_error"

    return True, "valid"


def make_valid_payload(record: dict[str, Any]) -> dict[str, Any]:
    output = get_output(record)
    assert output is not None

    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "file_source": record.get("file_source"),
        "sparql": output.get("sparql"),
        "execution_result": output.get("result"),
    }


def make_invalid_payload(record: dict[str, Any], reason: str) -> dict[str, Any]:
    output = get_output(record)

    payload = {
        "id": record.get("id"),
        "question": record.get("question"),
        "file_source": record.get("file_source"),
        "invalid_reason": reason,
        "status": record.get("status"),
        "returncode": record.get("returncode"),
        "error": record.get("error"),
        "raw_log": record.get("raw_log"),
        "per_question_json": record.get("per_question_json"),
    }

    if output is not None:
        payload["sparql"] = output.get("sparql")
        payload["execution_result"] = output.get("result")
        payload["answer"] = output.get("answer")
        payload["output_type"] = output.get("type")
    else:
        payload["sparql"] = None
        payload["execution_result"] = None
        payload["answer"] = None
        payload["output_type"] = None

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split GRASP all_results.json into valid and invalid directories."
    )
    parser.add_argument(
        "--input",
        default="output/Qwen3-30B-A3B-Instruct-2507/all_results.json",
        help="Path to all_results.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write split outputs. "
            "Default: same directory as all_results.json / split_results"
        ),
    )
    parser.add_argument(
        "--copy-raw-logs",
        action="store_true",
        help="Copy raw log files into valid/raw_logs and invalid/raw_logs if paths exist.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data = load_json(input_path)

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Input JSON must contain a list field named 'results'.")

    if args.output_dir is None:
        output_root = input_path.parent / "split_results"
    else:
        output_root = Path(args.output_dir)

    valid_dir = output_root / "valid"
    invalid_dir = output_root / "invalid"
    valid_json_dir = valid_dir / "json"
    invalid_json_dir = invalid_dir / "json"
    valid_raw_dir = valid_dir / "raw_logs"
    invalid_raw_dir = invalid_dir / "raw_logs"

    valid_json_dir.mkdir(parents=True, exist_ok=True)
    invalid_json_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_raw_logs:
        valid_raw_dir.mkdir(parents=True, exist_ok=True)
        invalid_raw_dir.mkdir(parents=True, exist_ok=True)

    valid_cases: list[dict[str, Any]] = []
    invalid_cases: list[dict[str, Any]] = []

    reason_counter = Counter()
    status_counter = Counter()

    for record in tqdm(results, desc="Splitting results", unit="case"):
        idx = record.get("id", "unknown")
        question = record.get("question") or ""
        slug = safe_slug(question)
        filename = f"{int(idx):04d}_{slug}.json" if isinstance(idx, int) else f"{idx}_{slug}.json"

        status_counter[str(record.get("status"))] += 1

        is_valid, reason = classify_record(record)
        reason_counter[reason] += 1

        if is_valid:
            payload = make_valid_payload(record)
            valid_cases.append(payload)
            out_path = valid_json_dir / filename
            write_json(out_path, payload)

            if args.copy_raw_logs:
                raw_log = record.get("raw_log")
                if raw_log:
                    raw_path = Path(raw_log)
                    if raw_path.exists():
                        shutil.copy2(raw_path, valid_raw_dir / raw_path.name)

        else:
            payload = make_invalid_payload(record, reason)
            invalid_cases.append(payload)
            out_path = invalid_json_dir / filename
            write_json(out_path, payload)

            if args.copy_raw_logs:
                raw_log = record.get("raw_log")
                if raw_log:
                    raw_path = Path(raw_log)
                    if raw_path.exists():
                        shutil.copy2(raw_path, invalid_raw_dir / raw_path.name)

    total = len(results)
    valid_count = len(valid_cases)
    invalid_count = len(invalid_cases)

    summary = {
        "input": str(input_path),
        "output_root": str(output_root),
        "total": total,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "valid_frequency": valid_count / total if total else 0,
        "invalid_frequency": invalid_count / total if total else 0,
        "status_counts": dict(status_counter),
        "reason_counts": dict(reason_counter),
        "valid_dir": str(valid_json_dir),
        "invalid_dir": str(invalid_json_dir),
    }

    write_json(output_root / "summary_stats.json", summary)
    write_json(output_root / "valid_all.json", valid_cases)
    write_json(output_root / "invalid_all.json", invalid_cases)

    print()
    print("Done.")
    print(f"Input: {input_path}")
    print(f"Output root: {output_root}")
    print(f"Total: {total}")
    print(f"Valid: {valid_count} ({summary['valid_frequency']:.2%})")
    print(f"Invalid: {invalid_count} ({summary['invalid_frequency']:.2%})")
    print()
    print("Invalid reason counts:")
    for reason, count in reason_counter.most_common():
        print(f"  {reason}: {count}")
    print()
    print(f"Valid per-case JSON: {valid_json_dir}")
    print(f"Invalid per-case JSON: {invalid_json_dir}")
    print(f"Combined valid JSON: {output_root / 'valid_all.json'}")
    print(f"Combined invalid JSON: {output_root / 'invalid_all.json'}")
    print(f"Summary JSON: {output_root / 'summary_stats.json'}")


if __name__ == "__main__":
    main()