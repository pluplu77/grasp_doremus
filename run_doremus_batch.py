#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm


def utc_now_iso() -> str:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def safe_slug(text: str, max_len: int = 90) -> str:
    """Create a filesystem-safe slug from a question."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "question"


def json_dump_to_file(obj: Any, path: Path) -> None:
    """Safely write Python object as JSON text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    GRASP usually prints logs plus a final one-line JSON object:
      {"type": "output", ...}

    This function tries to extract the last valid JSON object from stdout.
    """
    if not text:
        return None

    # Best case: final line is JSON.
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    # Fallback: find likely GRASP final JSON object.
    markers = ['{"type": "output"', '{"type":"output"']
    for marker in markers:
        idx = text.rfind(marker)
        if idx != -1:
            candidate = text[idx:].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    return None


def run_grasp(question: str, config_path: str, timeout: int, log_level: str) -> dict[str, Any]:
    """
    Run:
      echo "$question" | grasp --log-level INFO run configs/run_vllm.yaml
    """
    cmd = ["grasp", "--log-level", log_level, "run", config_path]

    proc = subprocess.run(
        cmd,
        input=question + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )

    parsed_output = extract_last_json_object(proc.stdout)

    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "parsed_output": parsed_output,
    }


def read_questions_csv(input_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV appears empty: {input_path}")

        if "question" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain a 'question' column. Found columns: {reader.fieldnames}"
            )

        for row in reader:
            question = (row.get("question") or "").strip()
            if not question:
                continue

            rows.append(
                {
                    "question": question,
                    "file_source": (row.get("file_source") or "").strip(),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GRASP on DOREMUS questions from a CSV file."
    )
    parser.add_argument(
        "--input",
        default="./data/doremus_question.csv",
        help="Input CSV path. Must contain a 'question' column.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/Qwen3-4B-Instruct-2507",
        help="Output directory.",
    )
    parser.add_argument(
        "--config",
        default="configs/run_vllm.yaml",
        help="GRASP config YAML.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per question in seconds.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="GRASP log level.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of questions to run.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    per_question_dir = output_dir / "per_question"
    raw_log_dir = output_dir / "raw_logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    per_question_dir.mkdir(parents=True, exist_ok=True)
    raw_log_dir.mkdir(parents=True, exist_ok=True)

    rows = read_questions_csv(input_path)

    if args.limit is not None:
        rows = rows[: args.limit]

    all_results: list[dict[str, Any]] = []

    for idx, row in enumerate(
        tqdm(rows, desc="Running GRASP", unit="question"),
        start=1,
    ):
        question = row["question"]
        file_source = row["file_source"]

        slug = safe_slug(question)
        base_name = f"{idx:04d}_{slug}"

        started_at = utc_now_iso()

        status = "unknown"
        error = None
        run_result: dict[str, Any]

        try:
            run_result = run_grasp(
                question=question,
                config_path=args.config,
                timeout=args.timeout,
                log_level=args.log_level,
            )

            status = "ok" if run_result["returncode"] == 0 else "error"

            # GRASP can return code 0 but still produce no parsed final JSON.
            if run_result["parsed_output"] is None:
                status = "no_parsed_output"

        except subprocess.TimeoutExpired as e:
            status = "timeout"
            error = f"Timed out after {args.timeout} seconds"
            stdout = e.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")

            run_result = {
                "command": ["grasp", "--log-level", args.log_level, "run", args.config],
                "returncode": None,
                "stdout": stdout,
                "parsed_output": None,
            }

        except Exception as e:
            status = "exception"
            error = repr(e)
            run_result = {
                "command": ["grasp", "--log-level", args.log_level, "run", args.config],
                "returncode": None,
                "stdout": "",
                "parsed_output": None,
            }

        finished_at = utc_now_iso()

        raw_log_path = raw_log_dir / f"{base_name}.txt"
        raw_log_path.write_text(
            run_result.get("stdout", "") or "",
            encoding="utf-8",
        )

        record: dict[str, Any] = {
            "id": idx,
            "question": question,
            "file_source": file_source,
            "status": status,
            "error": error,
            "returncode": run_result.get("returncode"),
            "started_at": started_at,
            "finished_at": finished_at,
            "command": run_result.get("command"),
            "parsed_output": run_result.get("parsed_output"),
            "per_question_json": str(per_question_dir / f"{base_name}.json"),
            "raw_log": str(raw_log_path),
        }

        per_question_path = per_question_dir / f"{base_name}.json"
        json_dump_to_file(record, per_question_path)

        all_results.append(record)

    combined = {
        "input": str(input_path),
        "config": args.config,
        "output_dir": str(output_dir),
        "num_questions": len(all_results),
        "created_at": utc_now_iso(),
        "environment": {
            "MODEL": os.environ.get("MODEL"),
            "MODEL_PROVIDER": os.environ.get("MODEL_PROVIDER"),
            "MODEL_ENDPOINT": os.environ.get("MODEL_ENDPOINT"),
            "KG_ENDPOINT": os.environ.get("KG_ENDPOINT"),
            "KG_NOTES_FILE": os.environ.get("KG_NOTES_FILE"),
        },
        "results": all_results,
    }

    combined_path = output_dir / "all_results.json"
    json_dump_to_file(combined, combined_path)

    print()
    print("Done.")
    print(f"Questions processed: {len(all_results)}")
    print(f"Per-question JSON: {per_question_dir}")
    print(f"Raw logs: {raw_log_dir}")
    print(f"Combined JSON: {combined_path}")


if __name__ == "__main__":
    main()