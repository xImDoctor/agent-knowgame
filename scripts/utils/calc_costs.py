"""Estimate API spend from token_usage.txt against a YAML prices table.

Reads token counts written by LLMClient._update_token_log and multiplies
them by per-1K-token prices loaded from a YAML file. 

Models without a matching price entry are reported as skipped.

Usage:
    python scripts/utils/calc_costs.py
    python scripts/utils/calc_costs.py --token-log token_usage.txt --prices configs/together_prices.yaml
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml


DEFAULT_TOKEN_LOG = Path('token_usage.txt')
DEFAULT_PRICES = Path('configs/together_prices.yaml')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Estimate API spend from token_usage.txt.')

    p.add_argument('--token-log', type=str, default=str(DEFAULT_TOKEN_LOG),
                   help=f'Path to token usage JSON (default: {DEFAULT_TOKEN_LOG})')
    p.add_argument('--prices', type=str, default=str(DEFAULT_PRICES),
                   help=f'Path to YAML pricing table (default: {DEFAULT_PRICES})')

    return p.parse_args()


def load_prices(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def main():
    args = parse_args()

    log_path = Path(args.token_log)
    if not log_path.exists():
        print(f'No token log found at {log_path}')
        return

    prices_path = Path(args.prices)
    if not prices_path.exists():
        print(f'No prices table found at {prices_path}')
        return

    totals = json.loads(log_path.read_text(encoding='utf-8'))
    prices = load_prices(prices_path)

    grand_total = 0.0

    print('Per-model spend estimate:')
    print('-' * 90)

    for key, tok in totals.items():
        prompt = int(tok.get('prompt_tokens', 0))
        completion = int(tok.get('completion_tokens', 0))
        entry = prices.get(key)

        if not entry:
            print(f'[WARN] No prices for {key}; tokens={prompt + completion} - skipped.')
            continue

        cost = (prompt / 1000.0) * entry['input'] + (completion / 1000.0) * entry['output']
        grand_total += cost
        print(f'{key:55s} prompt={prompt:8d}  completion={completion:8d}  cost=${cost:,.4f}')

    print('-' * 90)
    print(f'TOTAL SPEND: ${grand_total:,.4f}')


if __name__ == '__main__':
    main()
