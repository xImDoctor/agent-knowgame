"""The script that merges orphan token_usage part-files
for crashed runs into the right target.

Usage:
    python scripts/utils/merge_token_usage.py
    python scripts/utils/merge_token_usage.py --target token_usage.txt
    python scripts/utils/merge_token_usage.py --dry-run
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from game.clients.token_usage import merge_files, DEFAULT_VAL, PART_SUFFIX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Merge orphan token_usage .part files into canonical target.')

    p.add_argument('--target', type=str, default=str(DEFAULT_VAL),
                   help=f'Canonical token usage file (default: {DEFAULT_VAL})')

    p.add_argument('--dry-run', action='store_true',
                   help='List parts that would be merged; do nothing.')
    
    return p.parse_args()


def main():

    args = parse_args()
    target = Path(args.target)
    directory = target.parent if str(target.parent) else Path('.')
    parts = sorted(directory.glob(f'{target.stem}.*{PART_SUFFIX}'))

    if not parts:
        print(f'No {target.stem}.*{PART_SUFFIX} files found in {directory.resolve()}.')
        return

    print(f'Found {len(parts)} part file(s):')
    for p in parts:
        print(f'  {p}')

    if args.dry_run:
        print('\n[dry-run] Nothing merged.')
        return

    merged = merge_files(parts, target)
    print(f'\nMerged into {target}. Keys in target: {len(merged)}')


if __name__ == '__main__':
    main()
