"""Retry only error rows from probe JSONLs, writing a filled copy alongside.

Reads <input>.jsonl, identifies rows where `error != null`, reruns just those
cells through the same probe's run_one, and writes a new JSONL that contains
all successful rows from input PLUS the retry results (both successes and any
remaining errors). Also emits a matching CSV.

The input file is left untouched - the output is a separate `_retry_<ts>` file.

Auto-detects the probe type from the schema of input rows:
    "share"   - probe_share
    "request" - probe_request
    "number"  - probe_expected_rounds

Usage:
    python scripts/probes/retry_probe_errors.py probes/probe_share_qwen_2026-07-31.jsonl
    python scripts/probes/retry_probe_errors.py <input> --output <out> --request-timeout 240
    python scripts/probes/retry_probe_errors.py <input> --payment-mode student_pays

"""

import sys
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

from game.clients.token_usage import token_usage_session

# sibling probes importable coz the same sys.path
import probe_share
import probe_request
import probe_expected_rounds


# result-field-name to probe module
PROBE_BY_FIELD = {
    'share':   probe_share,
    'request': probe_request,
    'number':  probe_expected_rounds,
}


def detect_probe_module(rows: list[dict]) -> tuple:
    if not rows:
        raise ValueError('input JSONL is empty')

    keys = set(rows[0].keys())
    matches = [name for name in PROBE_BY_FIELD if name in keys]

    if len(matches) != 1:
        raise ValueError(f'cannot uniquely identify probe type: got {matches} in row keys')

    field = matches[0]
    return PROBE_BY_FIELD[field], field


def parse_bool(v: str) -> bool:
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    if v.lower() in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'expected true/false, got {v!r}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Retry error rows from a probe JSONL. Writes a filled copy alongside.'
    )

    p.add_argument('input', type=str,
                   help='Path to input JSONL from a prior probe run')
    p.add_argument('--output', type=str, default=None,
                   help='Output JSONL path. Default: <input_stem>_retry_<YYYYMMDD-HHMMSS>.jsonl next to input')
    p.add_argument('--reasoning', type=parse_bool, default=None,
                   help='Force reasoning on/off. Default: auto-detect from successful input rows')
    p.add_argument('--request-timeout', type=float, default=60.0,
                   help='LLM call timeout, seconds (default 60)')
    p.add_argument('--payment-mode', type=str, default='teacher_pays',
                   choices=['teacher_pays', 'student_pays'],
                   help='Only used for probe_expected_rounds input (not logged in per-row schema). Default teacher_pays')

    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def default_output_path(input_path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    return input_path.parent / f'{input_path.stem}_retry_{ts}.jsonl'

# reconstruct stub_config from a JSONL row (per-probe stub signature)
def build_stub_config(probe_module, row: dict, request_timeout: float, payment_mode: str):

    common = dict(
        n_agents=int(row['n_agents']),
        m_informed=int(row['m_informed']),
        share_cost=float(row['share_cost']),
        seed=int(row['seed']),
        model=str(row['model']),
        api_type=str(row['api_type']),
        request_timeout=request_timeout,
    )

    # only probe_expected_rounds.stub_config accepts payment_mode
    if probe_module is probe_expected_rounds:
        common['payment_mode'] = payment_mode

    return probe_module.stub_config(**common)


def write_csv(jsonl_path: Path, csv_path: Path, result_field: str) -> None:
    rows = read_jsonl(jsonl_path)

    if not rows:
        return

    fields = ['ts', 'model', 'api_type', 'n_agents', 'm_informed', 'share_cost',
              'seed', result_field, 'reasoning', 'error']

    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():

    load_dotenv()

    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f'input not found: {input_path}')

    rows = read_jsonl(input_path)
    probe_module, result_field = detect_probe_module(rows)

    good_rows  = [r for r in rows if r.get('error') is None]
    error_rows = [r for r in rows if r.get('error') is not None]

    if not error_rows:
        print(f'No error rows in {input_path}. Nothing to retry.')
        return

    # auto-detect: reasoning if at least one good row has a non-null reasoning field
    if args.reasoning is None:
        use_reasoning = any(r.get('reasoning') is not None for r in good_rows)
    else:
        use_reasoning = args.reasoning

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix('.csv')

    print(f'Probe type:   {probe_module.__name__} (field={result_field!r})')
    print(f'Input rows:   {len(rows)} total = {len(good_rows)} ok + {len(error_rows)} errors')
    print(f'Reasoning:    {use_reasoning} ({"auto-detected" if args.reasoning is None else "forced"})')
    print(f'Timeout:      {args.request_timeout}s')

    if probe_module is probe_expected_rounds:
        print(f'Payment mode: {args.payment_mode}')

    print(f'Writing to:   {output_path}')

    total = len(error_rows)
    still_errored = 0

    with token_usage_session() as tok, output_path.open('w', encoding='utf-8') as f:
        # carry over all good rows unchanged
        for r in good_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        f.flush()

        # retry each error row through the same probe's run_one
        for i, r in enumerate(error_rows, 1):
            config = build_stub_config(probe_module, r, args.request_timeout, args.payment_mode)
            record = probe_module.run_one(config, use_reasoning, token_log_path=tok)

            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()

            if record.get('error') is not None:
                still_errored += 1

            print(f'  [{i}/{total}] '
                  f'n={record["n_agents"]:>2} k={record["m_informed"]:>2} '
                  f'p={record["share_cost"]:<5} seed={record["seed"]:>3} '
                  f'{result_field}={record.get(result_field)} err={record["error"]}')

    write_csv(output_path, csv_path, result_field)

    print(f'\nRetried {total} rows; {still_errored} still errored.')
    print(f'JSONL: {output_path}')
    print(f'CSV:   {csv_path}')


if __name__ == '__main__':
    main()

