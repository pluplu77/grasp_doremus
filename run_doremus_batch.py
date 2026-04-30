#!/usr/bin/env python3
"""
chmod +x run_doremus_batch.py

export MODEL="Qwen/Qwen3-4B-Instruct-2507"
export MODEL_PROVIDER="openai/completions"
export MODEL_ENDPOINT="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
export KG_ENDPOINT="https://data.doremus.org/sparql"

unset KG_NOTES_FILE

./run_doremus_batch.py \
  --input ./data/doremus_question.csv \
  --output-dir output/Qwen3-4B-Instruct-2507 \
  --config configs/run_vllm.yaml
  
"""

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime


def safe_slug(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "question"


def extract_last_json(stdout: str):
    """
    GRASP usually prints logs plus a final JSON object.
    This extracts the last valid JSON object from stdout.
    """
    lines = stdout.strip().splitlines()

    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Fallback: try from the last occurrence of {"type":
    idx = stdout.rfind('{"type"')
    if idx != -1:
        candidate = stdout[idx:].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def run_grasp(question: str, config_path: str, timeout: int):
    cmd = ["grasp", "--log-level", "INFO", "run", config_path]

    proc = subprocess.run(
        cmd,
        input=question + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )

    parsed = extract_last_json(proc.stdout)

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "parsed_output": parsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="./data/doremus_question.csv",
        help="CSV file with a question column.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/Qwen3-4B-Instruct-2507",
        help="Directory where JSON outputs will be written.",
    )
    parser.add_argument(
        "--config",
        default="configs/run_vllm.yaml",
        help="GRASP run config YAML.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per question in seconds.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    per_question_dir = output_dir / "per_question"
    raw_log_dir = output_dir / "raw_logs"

    per_question_dir.mkdir(parents=True, exist_ok=True)
    raw_log_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "question" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain a 'question' column. Found columns: {reader.fieldnames}"
            )

        for idx, row in enumerate(reader, start=1):
            question = row["question"].strip()
            file_source = row.get("file_source", "")

            if not question:
                continue

            slug = safe_slug(question)
            base_name = f"{idx:04d}_{slug}"

            print(f"[{idx}] Running: {question}")

            started_at = datetime.utcnow().isoformat() + "Z"

            try:
                run = run_grasp(question, args.config, args.timeout)
                status = "ok" if run["returncode"] == 0 else "error"
                error = None
            except subprocess.TimeoutExpired as e:
                run = {
                    "returncode": None,
                    "stdout": e.stdout or "",
                    "parsed_output": None,
                }
                status = "timeout"
                error = f"Timed out after {args.timeout} seconds"
            except Exception as e:
                run = {
                    "returncode": None,
                    "stdout": "",
                    "parsed_output": None,
                }
                status = "exception"
                error = repr(e)

            finished_at = datetime.utcnow().isoformat() + "Z"

            record = {
                "id": idx,
                "question": question,
                "file_source": file_source,
                "status": status,
                "error": error,
                "returncode": run["returncode"],
                "started_at": started_at,
                "finished_at": finished_at,
                "parsed_output": run["parsed_output"],
            }

            # Save raw stdout/log separately for debugging.
            raw_log_path = raw_log_dir / f"{base_name}.txt"
            raw_log_path.write_text(run["stdout"], encoding="utf-8")

            # Save per-question JSON.
            per_question_path = per_question_dir / f"{base_name}.json"
            per_question_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            record["per_question_json"] = str(per_question_path)
            record["raw_log"] = str(raw_log_path)

            all_results.append(record)

    combined = {
        "input": str(input_path),
        "config": args.config,
        "output_dir": str(output_dir),
        "num_questions": len(all_results),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "results": all_results,
    }

    combined_path = output_dir / "all_results.json"
    combined_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone.")
    print(f"Per-question JSON: {per_question_dir}")
    print(f"Raw logs: {raw_log_dir}")
    print(f"Combined JSON: {combined_path}")


if __name__ == "__main__":
    main()