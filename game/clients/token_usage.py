"""
Cross-process safe token-usage accumulation
for parallel script/game runs

On clean session exit the part is atomically
merged into the canonical target under a file lock.
On any exception (including KeyboardInterrupt) the 
part is left in place for manual merge
via scripts/merge_token_usage.py
"""

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock

DEFAULT_VAL = Path('token_usage.txt')
PART_SUFFIX = '.part'
LOCK_TIMEOUT_SEC = 60.0


def _load(path: Path) -> dict:

    if not path.exists():
        return {}

    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return {}

    return json.loads(text)


def _add_into(dst: dict, src: dict) -> None:
    for key, tok in src.items():
        entry = dst.get(key, {'prompt_tokens': 0, 'completion_tokens': 0})
        entry['prompt_tokens'] += int(tok.get('prompt_tokens', 0) or 0)
        entry['completion_tokens'] += int(tok.get('completion_tokens', 0) or 0)

        dst[key] = entry


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.new')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')

    os.replace(tmp, path)


def merge_files(sources: list[Path], target: Path = DEFAULT_VAL) -> dict:

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(target) + '.lock', timeout=LOCK_TIMEOUT_SEC)

    with lock:
        merged = _load(target)
        loaded_sources: list[Path] = []

        for src in sources:
            src = Path(src)

            if not src.exists():
                continue
            data = _load(src)

            if data:
                _add_into(merged, data)

            loaded_sources.append(src)

        _atomic_write(target, merged)

        for src in loaded_sources:
            try:
                src.unlink()
            except OSError:
                pass

    return merged


def _new_part_path(target: Path) -> Path:

    ts = time.strftime('%Y%m%d-%H%M%S')
    unique = uuid.uuid4().hex[:6]

    return target.parent / f'{target.stem}.{os.getpid()}.{ts}-{unique}{PART_SUFFIX}'


# todo: change to Generator[Foo] instead
@contextmanager
def token_usage_session(target: Path = DEFAULT_VAL) -> Iterator[Path]:

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = _new_part_path(target)

    yield part_path
    # Merge below runs only on clean exit
    # @contextmanager skips
    # this section if an exception propagated through yield

    if part_path.exists():
        merge_files([part_path], target)

