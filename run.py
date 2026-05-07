#!/usr/bin/env python3

import csv
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional


# =========================
# Fixed paths / settings
# =========================

INPUT_CSV = Path("./data/doremus_question.csv")
OUTPUT_DIR = Path("output/Qwen3-30B-A3B-Instruct-2507")
CONFIG_PATH = "configs/run_vllm.yaml"

TIMEOUT_SECONDS = 900
LOG_LEVEL = "DEBUG"

JSON_DIR = OUTPUT_DIR / "json"
LOG_DIR = OUTPUT_DIR / "log"
ALL_RESULTS_PATH = OUTPUT_DIR / "all_results.json"


# =========================
# Logging helpers
# =========================

def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def local_now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def console(message: str = "") -> None:
    print(message, flush=True)


def write_log_line(log_file, line: str) -> None:
    log_file.write(line)
    log_file.flush()


def write_step(log_file, message: str) -> None:
    line = f"[{local_now_str()}] {message}\n"
    print(line, end="", flush=True)
    write_log_line(log_file, line)


# =========================
# File helpers
# =========================

def safe_question_filename(question: str, max_len: int = 180) -> str:
    name = question.strip()
    name = re.sub(r'[\\/:"*<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:max_len] or "question"


def make_base_name(idx: int, question: str) -> str:
    """
    Stable resume-safe name.

    Example:
    0001_Which works have been composed by Mozart ?.json
    """
    return f"{idx:04d}_{safe_question_filename(question)}"


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


# =========================
# GRASP JSON parsing
# =========================

def extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    Extracts the last JSON object from GRASP output.

    GRASP DEBUG logs may print many lines before the final JSON.
    """
    if not text:
        return None

    # Try complete JSON object on a single line.
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    # Try known GRASP output markers.
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

    # Fallback: scan backward from opening braces.
    last_open = text.rfind("{")
    while last_open != -1:
        candidate = text[last_open:].strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            last_open = text.rfind("{", 0, last_open)

    return None


# =========================
# CSV reader
# =========================

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
                f"CSV must contain a 'question' column. Found columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()

            if not question:
                continue

            rows.append(
                {
                    "csv_row_number": str(row_number),
                    "question": question,
                    "file_source": (row.get("file_source") or "").strip(),
                }
            )

    return rows


# =========================
# Resume/completion logic
# =========================

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
    """
    Resume rule:
    only files with parsed_output.output.sparql are considered complete.
    """
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
            return "cancel"

        return f"output_type:{output_type}"

    return "invalid_output_format"


# =========================
# Live GRASP runner
# =========================

def run_grasp_live(question: str, log_path: Path) -> dict[str, Any]:
    """
    Runs GRASP and streams output live.

    Important behavior:
    - prints GRASP stdout/stderr immediately to terminal
    - writes every line immediately to log_path
    - still stores full output text for JSON parsing at the end
    """
    cmd = ["grasp", "--log-level", LOG_LEVEL, "run", CONFIG_PATH]

    output_chunks: list[str] = []
    started_monotonic = time.monotonic()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        write_step(log_file, "============================================================")
        write_step(log_file, "Starting question")
        write_step(log_file, f"Question: {question}")
        write_step(log_file, f"Command: {' '.join(cmd)}")
        write_step(log_file, f"Config: {CONFIG_PATH}")
        write_step(log_file, f"Log level: {LOG_LEVEL}")
        write_step(log_file, "Launching GRASP process")
        write_step(log_file, "============================================================")

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdin is not None
        assert proc.stdout is not None

        proc.stdin.write(question + "\n")
        proc.stdin.flush()
        proc.stdin.close()

        timed_out = False

        try:
            while True:
                if proc.poll() is not None:
                    # Drain remaining output after process exits.
                    remaining = proc.stdout.read()
                    if remaining:
                        print(remaining, end="", flush=True)
                        write_log_line(log_file, remaining)
                        output_chunks.append(remaining)
                    break

                if time.monotonic() - started_monotonic > TIMEOUT_SECONDS:
                    timed_out = True
                    write_step(
                        log_file,
                        f"TIMEOUT reached after {TIMEOUT_SECONDS} seconds. Killing GRASP process.",
                    )
                    proc.kill()

                    remaining = proc.stdout.read()
                    if remaining:
                        print(remaining, end="", flush=True)
                        write_log_line(log_file, remaining)
                        output_chunks.append(remaining)

                    break

                line = proc.stdout.readline()

                if line:
                    # This is the key part:
                    # live terminal print + immediate log write.
                    print(line, end="", flush=True)
                    write_log_line(log_file, line)
                    output_chunks.append(line)
                else:
                    time.sleep(0.05)

        finally:
            returncode = proc.wait()

        elapsed = time.monotonic() - started_monotonic
        stdout = "".join(output_chunks)

        write_step(log_file, "============================================================")
        write_step(log_file, "GRASP process finished")
        write_step(log_file, f"Return code: {returncode}")
        write_step(log_file, f"Elapsed seconds: {elapsed:.2f}")
        write_step(log_file, f"Captured output characters: {len(stdout)}")
        write_step(log_file, "============================================================")

    parsed_output = extract_last_json_object(stdout)

    return {
        "command": cmd,
        "returncode": returncode,
        "stdout": stdout,
        "parsed_output": parsed_output,
        "elapsed_seconds": elapsed,
        "timed_out": timed_out,
    }


# =========================
# Per-question runner
# =========================

def build_record(
    idx: int,
    row: dict[str, str],
    run_result: dict[str, Any],
    status: str,
    error: Optional[str],
    started_at: str,
    finished_at: str,
    json_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    return {
        "id": idx,
        "csv_row_number": row.get("csv_row_number"),
        "question": row["question"],
        "file_source": row.get("file_source", ""),
        "status": status,
        "error": error,
        "returncode": run_result.get("returncode"),
        "elapsed_seconds": run_result.get("elapsed_seconds"),
        "started_at": started_at,
        "finished_at": finished_at,
        "command": run_result.get("command"),
        "parsed_output": run_result.get("parsed_output"),
        "json_path": str(json_path),
        "log_path": str(log_path),
    }


def run_one_question(idx: int, row: dict[str, str]) -> dict[str, Any]:
    question = row["question"]
    base_name = make_base_name(idx, question)

    json_path = JSON_DIR / f"{base_name}.json"
    log_path = LOG_DIR / f"{base_name}.log"

    console()
    console("############################################################")
    console(f"QUESTION {idx}")
    console(f"CSV row: {row.get('csv_row_number')}")
    console(f"Question: {question}")
    console(f"JSON: {json_path}")
    console(f"Log:  {log_path}")
    console("############################################################")

    started_at = utc_now_iso()
    error = None

    try:
        run_result = run_grasp_live(question, log_path)

        if run_result.get("timed_out"):
            status = "timeout"
            error = f"Timed out after {TIMEOUT_SECONDS} seconds"
        else:
            status = classify_status(run_result)

    except Exception as e:
        run_result = {
            "command": ["grasp", "--log-level", LOG_LEVEL, "run", CONFIG_PATH],
            "returncode": None,
            "stdout": "",
            "parsed_output": None,
            "elapsed_seconds": None,
            "timed_out": False,
        }
        status = "exception"
        error = repr(e)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
            write_step(log_file, f"EXCEPTION: {error}")

    finished_at = utc_now_iso()

    record = build_record(
        idx=idx,
        row=row,
        run_result=run_result,
        status=status,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        json_path=json_path,
        log_path=log_path,
    )

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    write_json(json_path, record)

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        write_step(log_file, f"Final status: {status}")
        write_step(log_file, f"Has SPARQL: {has_sparql(record)}")
        write_step(log_file, f"Wrote JSON: {json_path}")

    console()
    console(f"Finished question {idx}")
    console(f"Status: {status}")
    console(f"Has SPARQL: {has_sparql(record)}")
    console(f"JSON saved: {json_path}")
    console(f"Log saved:  {log_path}")

    return record


# =========================
# Resume scan
# =========================

def scan_existing_outputs(
    questions: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, str], str]]]:
    existing_records: list[dict[str, Any]] = []
    rerun_items: list[tuple[int, dict[str, str], str]] = []
    counts: dict[str, int] = {}

    console()
    console("Existing output scan")
    console("====================")

    for idx, row in enumerate(questions, start=1):
        base_name = make_base_name(idx, row["question"])
        json_path = JSON_DIR / f"{base_name}.json"

        record = read_json(json_path) if json_path.exists() else None
        classification = classify_record(record)

        counts[classification] = counts.get(classification, 0) + 1

        if record is not None:
            existing_records.append(record)

        if classification != "has_sparql":
            rerun_items.append((idx, row, classification))

    total = len(questions)
    existing_files = len(list(JSON_DIR.glob("*.json"))) if JSON_DIR.exists() else 0
    has_sparql_count = counts.get("has_sparql", 0)
    no_sparql_count = total - has_sparql_count

    console(f"Input questions: {total}")
    console(f"JSON files in output directory: {existing_files}")
    console(f"Questions with generated SPARQL: {has_sparql_count}")
    console(f"Questions missing/no SPARQL: {no_sparql_count}")
    console()
    console("Breakdown:")
    for key, value in sorted(counts.items()):
        console(f"  {key}: {value}")
    console()

    return existing_records, rerun_items


# =========================
# Combined result rebuild
# =========================

def rebuild_all_results(questions: list[dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(questions, start=1):
        base_name = make_base_name(idx, row["question"])
        json_path = JSON_DIR / f"{base_name}.json"
        log_path = LOG_DIR / f"{base_name}.log"

        record = read_json(json_path)

        if record is not None:
            records.append(record)
        else:
            records.append(
                {
                    "id": idx,
                    "csv_row_number": row.get("csv_row_number"),
                    "question": row["question"],
                    "file_source": row.get("file_source", ""),
                    "status": "missing_json",
                    "error": f"No per-question JSON found at {json_path}",
                    "returncode": None,
                    "elapsed_seconds": None,
                    "started_at": None,
                    "finished_at": None,
                    "command": ["grasp", "--log-level", LOG_LEVEL, "run", CONFIG_PATH],
                    "parsed_output": None,
                    "json_path": str(json_path),
                    "log_path": str(log_path),
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
        "input_csv": str(INPUT_CSV),
        "config_path": CONFIG_PATH,
        "output_dir": str(OUTPUT_DIR),
        "json_dir": str(JSON_DIR),
        "log_dir": str(LOG_DIR),
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

    console()
    console(f"Wrote combined results: {ALL_RESULTS_PATH}")
    console("Status counts:")
    for key, value in sorted(status_counts.items()):
        console(f"  {key}: {value}")


# =========================
# Prompt
# =========================

def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()

        if ans in {"y", "yes"}:
            return True

        if ans in {"n", "no"}:
            return False

        print("Please type yes or no.")


# =========================
# Main
# =========================

def main() -> None:
    console("Starting GRASP batch runner")
    console(f"Input CSV: {INPUT_CSV}")
    console(f"Output directory: {OUTPUT_DIR}")
    console(f"JSON directory: {JSON_DIR}")
    console(f"Log directory: {LOG_DIR}")
    console(f"Config path: {CONFIG_PATH}")
    console(f"Log level: {LOG_LEVEL}")
    console(f"Timeout seconds: {TIMEOUT_SECONDS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    console()
    console("Reading CSV...")
    questions = read_questions(INPUT_CSV)
    console(f"Loaded {len(questions)} questions.")

    _, rerun_items = scan_existing_outputs(questions)

    if not rerun_items:
        console("All questions already have generated SPARQL.")
        console("Rebuilding all_results.json only.")
        write_combined_results(questions)
        console("Done.")
        return

    console(f"Resume found {len(rerun_items)} questions that need rerun.")

    should_rerun = ask_yes_no(
        f"Do you want to rerun the {len(rerun_items)} questions without generated SPARQL? [yes/no]: "
    )

    if not should_rerun:
        console("No rerun performed.")
        console("Rebuilding all_results.json from existing files.")
        write_combined_results(questions)
        console("Done.")
        return

    total_rerun = len(rerun_items)

    for rerun_index, (idx, row, reason) in enumerate(rerun_items, start=1):
        console()
        console("============================================================")
        console(f"Rerun item {rerun_index}/{total_rerun}")
        console(f"Original question id: {idx}/{len(questions)}")
        console(f"Rerun reason: {reason}")
        console("============================================================")

        run_one_question(idx, row)

    write_combined_results(questions)

    console()
    console("Final scan after rerun:")
    scan_existing_outputs(questions)

    console()
    console("Done.")
    console(f"Input: {INPUT_CSV}")
    console(f"Output directory: {OUTPUT_DIR}")
    console(f"Per-question JSON: {JSON_DIR}")
    console(f"Raw logs: {LOG_DIR}")
    console(f"Combined JSON: {ALL_RESULTS_PATH}")


if __name__ == "__main__":
    main()