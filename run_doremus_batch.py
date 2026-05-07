#!/usr/bin/env python3

import csv
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =========================
# Fixed paths / settings
# =========================

INPUT_CSV = Path("./data/doremus_question.csv")
OUTPUT_DIR = Path("output/Qwen3-30B-A3B-Instruct-2507")
CONFIG_PATH = "configs/run_vllm.yaml"

TIMEOUT_SECONDS = 900
LOG_LEVEL = "DEBUG"

PER_QUESTION_DIR = OUTPUT_DIR / "per_question"
RAW_LOG_DIR = OUTPUT_DIR / "raw_logs"
ALL_RESULTS_PATH = OUTPUT_DIR / "all_results.json"


# =========================
# Helpers
# =========================

def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def safe_slug(text: str, max_len: int = 100) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "question"


def make_base_name(idx: int, question: str) -> str:
    return f"{idx:04d}_{safe_slug(question)}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None

    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

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
                continue

    return None


def read_questions(input_csv: Path) -> list[dict[str, str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows: list[dict[str, str]] = []

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV is empty: {input_csv}")

        if "question" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain a 'question' column. Found: {reader.fieldnames}"
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


def get_output_obj(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    parsed = record.get("parsed_output")
    if not isinstance(parsed, dict):
        return None

    output = parsed.get("output")
    if not isinstance(output, dict):
        return None

    return output


def has_sparql(record: Optional[dict[str, Any]]) -> bool:
    if not isinstance(record, dict):
        return False

    output = get_output_obj(record)
    if not isinstance(output, dict):
        return False

    sparql = output.get("sparql")
    return isinstance(sparql, str) and bool(sparql.strip())


def classify_record(record: Optional[dict[str, Any]]) -> str:
    if record is None:
        return "missing_json"

    if has_sparql(record):
        return "has_sparql"

    status = record.get("status")
    if status:
        return f"no_sparql:{status}"

    parsed = record.get("parsed_output")
    if not isinstance(parsed, dict):
        return "no_sparql:missing_parsed_output"

    output = parsed.get("output")
    if output is None:
        return "no_sparql:null_output"

    return "no_sparql:other"


def run_grasp(question: str) -> dict[str, Any]:
    cmd = ["grasp", "--log-level", LOG_LEVEL, "run", CONFIG_PATH]

    started = time.monotonic()

    proc = subprocess.run(
        cmd,
        input=question + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )

    elapsed = time.monotonic() - started
    stdout = proc.stdout or ""

    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": stdout,
        "parsed_output": extract_last_json_object(stdout),
        "elapsed_seconds": elapsed,
    }


def classify_status(run_result: dict[str, Any]) -> str:
    if run_result.get("returncode") != 0:
        return "error"

    parsed = run_result.get("parsed_output")
    if not isinstance(parsed, dict):
        return "no_parsed_output"

    output = parsed.get("output")
    if output is None:
        return "missing_or_null_output"

    if isinstance(output, dict):
        output_type = output.get("type")

        if output_type == "answer":
            result_text = str(output.get("result") or "").lower()
            if (
                "sparql execution failed" in result_text
                or "error executing sparql" in result_text
                or "server error" in result_text
                or "timed out" in result_text
                or "failed to parse input" in result_text
            ):
                return "execution_error"
            return "valid"

        if output_type == "cancel":
            # A cancel can still contain a useful SPARQL. We keep it,
            # but status says cancel.
            return "cancel"

        return f"output_type:{output_type}"

    return "invalid_output_format"


def build_record(
    idx: int,
    question: str,
    file_source: str,
    run_result: dict[str, Any],
    status: str,
    error: Optional[str],
    started_at: str,
    finished_at: str,
    per_question_path: Path,
    raw_log_path: Path,
) -> dict[str, Any]:
    return {
        "id": idx,
        "question": question,
        "file_source": file_source,
        "status": status,
        "error": error,
        "returncode": run_result.get("returncode"),
        "elapsed_seconds": run_result.get("elapsed_seconds"),
        "started_at": started_at,
        "finished_at": finished_at,
        "command": run_result.get("command"),
        "parsed_output": run_result.get("parsed_output"),
        "per_question_json": str(per_question_path),
        "raw_log": str(raw_log_path),
    }


def run_one_question(idx: int, row: dict[str, str]) -> dict[str, Any]:
    question = row["question"]
    file_source = row["file_source"]
    base_name = make_base_name(idx, question)

    per_question_path = PER_QUESTION_DIR / f"{base_name}.json"
    raw_log_path = RAW_LOG_DIR / f"{base_name}.txt"

    started_at = utc_now_iso()

    try:
        run_result = run_grasp(question)
        error = None
        status = classify_status(run_result)

    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")

        run_result = {
            "command": ["grasp", "--log-level", LOG_LEVEL, "run", CONFIG_PATH],
            "returncode": None,
            "stdout": stdout,
            "parsed_output": extract_last_json_object(stdout),
            "elapsed_seconds": TIMEOUT_SECONDS,
        }
        status = "timeout"
        error = f"Timed out after {TIMEOUT_SECONDS} seconds"

    except Exception as e:
        run_result = {
            "command": ["grasp", "--log-level", LOG_LEVEL, "run", CONFIG_PATH],
            "returncode": None,
            "stdout": "",
            "parsed_output": None,
            "elapsed_seconds": None,
        }
        status = "exception"
        error = repr(e)

    finished_at = utc_now_iso()

    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    raw_log_path.write_text(run_result.get("stdout", "") or "", encoding="utf-8")

    record = build_record(
        idx=idx,
        question=question,
        file_source=file_source,
        run_result=run_result,
        status=status,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        per_question_path=per_question_path,
        raw_log_path=raw_log_path,
    )

    PER_QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    write_json(per_question_path, record)

    return record


def scan_existing_outputs(
    questions: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, str], str]]]:
    existing_records: list[dict[str, Any]] = []
    rerun_items: list[tuple[int, dict[str, str], str]] = []

    counts: dict[str, int] = {}

    for idx, row in enumerate(questions, start=1):
        base_name = make_base_name(idx, row["question"])
        per_question_path = PER_QUESTION_DIR / f"{base_name}.json"

        record = read_json(per_question_path) if per_question_path.exists() else None
        classification = classify_record(record)

        counts[classification] = counts.get(classification, 0) + 1

        if record is not None:
            existing_records.append(record)

        if classification != "has_sparql":
            rerun_items.append((idx, row, classification))

    total = len(questions)
    existing_files = len(list(PER_QUESTION_DIR.glob("*.json"))) if PER_QUESTION_DIR.exists() else 0
    has_sparql_count = counts.get("has_sparql", 0)
    no_sparql_count = total - has_sparql_count

    print()
    print("Existing output scan")
    print("====================")
    print(f"Input questions: {total}")
    print(f"JSON files in output directory: {existing_files}")
    print(f"Questions with generated SPARQL: {has_sparql_count}")
    print(f"Questions missing/no SPARQL: {no_sparql_count}")
    print()
    print("Breakdown:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    print()

    return existing_records, rerun_items


def rebuild_all_results(questions: list[dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(questions, start=1):
        base_name = make_base_name(idx, row["question"])
        per_question_path = PER_QUESTION_DIR / f"{base_name}.json"
        record = read_json(per_question_path)
        if record is not None:
            records.append(record)
        else:
            records.append(
                {
                    "id": idx,
                    "question": row["question"],
                    "file_source": row["file_source"],
                    "status": "missing_json",
                    "error": f"No per-question JSON found at {per_question_path}",
                    "returncode": None,
                    "elapsed_seconds": None,
                    "started_at": None,
                    "finished_at": None,
                    "command": ["grasp", "--log-level", LOG_LEVEL, "run", CONFIG_PATH],
                    "parsed_output": None,
                    "per_question_json": str(per_question_path),
                    "raw_log": str(RAW_LOG_DIR / f"{base_name}.txt"),
                }
            )

    return records


def write_combined_results(questions: list[dict[str, str]]) -> None:
    records = rebuild_all_results(questions)

    status_counts: dict[str, int] = {}
    sparql_count = 0

    for record in records:
        status = str(record.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if has_sparql(record):
            sparql_count += 1

    combined = {
        "input": str(INPUT_CSV),
        "config": CONFIG_PATH,
        "output_dir": str(OUTPUT_DIR),
        "log_level": LOG_LEVEL,
        "timeout_seconds": TIMEOUT_SECONDS,
        "num_questions": len(records),
        "num_with_sparql": sparql_count,
        "num_without_sparql": len(records) - sparql_count,
        "created_at": utc_now_iso(),
        "status_counts": status_counts,
        "results": records,
    }

    write_json(ALL_RESULTS_PATH, combined)


def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please type yes or no.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)

    questions = read_questions(INPUT_CSV)

    _, rerun_items = scan_existing_outputs(questions)

    if not rerun_items:
        print("All questions already have generated SPARQL. Rebuilding all_results.json only.")
        write_combined_results(questions)
        print(f"Combined JSON: {ALL_RESULTS_PATH}")
        return

    should_rerun = ask_yes_no(
        f"Do you want to rerun the {len(rerun_items)} questions without generated SPARQL? [yes/no]: "
    )

    if not should_rerun:
        print("No rerun performed. Rebuilding all_results.json from existing files.")
        write_combined_results(questions)
        print(f"Combined JSON: {ALL_RESULTS_PATH}")
        return

    iterator = rerun_items
    if tqdm is not None:
        iterator = tqdm(
            rerun_items,
            total=len(rerun_items),
            desc="Rerunning no-SPARQL cases",
            unit="question",
        )

    for idx, row, reason in iterator:
        if tqdm is None:
            print(f"Rerunning {idx}: {row['question']}  ({reason})")

        record = run_one_question(idx, row)

        if tqdm is None:
            print(f"  -> status={record.get('status')}, has_sparql={has_sparql(record)}")

    write_combined_results(questions)

    print()
    print("Done.")
    print(f"Input: {INPUT_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Per-question JSON: {PER_QUESTION_DIR}")
    print(f"Raw logs: {RAW_LOG_DIR}")
    print(f"Combined JSON: {ALL_RESULTS_PATH}")

    # Final scan after rerun.
    scan_existing_outputs(questions)


if __name__ == "__main__":
    main()