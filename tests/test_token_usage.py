import json
import threading
from pathlib import Path

import pytest

from game.clients.token_usage import merge_files, token_usage_session


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding='utf-8')


def test_merge_two_sources_into_empty_target(tmp_path):
    target = tmp_path / 'token_usage.txt'
    a = tmp_path / 'token_usage.a.part'
    b = tmp_path / 'token_usage.b.part'

    _write(a, {'together:m1': {'prompt_tokens': 100, 'completion_tokens': 20}})
    _write(b, {'together:m1': {'prompt_tokens': 50, 'completion_tokens': 10},
               'together:m2': {'prompt_tokens': 5, 'completion_tokens': 1}})

    merge_files([a, b], target)

    result = json.loads(target.read_text(encoding='utf-8'))

    assert result == {
        'together:m1': {'prompt_tokens': 150, 'completion_tokens': 30},
        'together:m2': {'prompt_tokens': 5, 'completion_tokens': 1},
    }
    assert not a.exists() and not b.exists()


def test_merge_adds_to_existing_target(tmp_path):
    target = tmp_path / 'token_usage.txt'
    _write(target, {'together:m1': {'prompt_tokens': 1000, 'completion_tokens': 100}})

    src = tmp_path / 'token_usage.a.part'
    _write(src, {'together:m1': {'prompt_tokens': 5, 'completion_tokens': 2}})

    merge_files([src], target)

    result = json.loads(target.read_text(encoding='utf-8'))
    assert result == {'together:m1': {'prompt_tokens': 1005, 'completion_tokens': 102}}


def test_session_merges_on_clean_exit(tmp_path):
    target = tmp_path / 'token_usage.txt'

    with token_usage_session(target) as part:
        _write(part, {'together:m1': {'prompt_tokens': 42, 'completion_tokens': 7}})
        assert part.exists()

    assert not part.exists()
    result = json.loads(target.read_text(encoding='utf-8'))
    assert result == {'together:m1': {'prompt_tokens': 42, 'completion_tokens': 7}}


def test_session_preserves_part_on_exception(tmp_path):
    target = tmp_path / 'token_usage.txt'
    saved: list[Path] = []

    with pytest.raises(RuntimeError, match='boom'):
        with token_usage_session(target) as part:
            _write(part, {'together:m1': {'prompt_tokens': 42, 'completion_tokens': 7}})
            saved.append(part)
            raise RuntimeError('boom')

    assert saved[0].exists(), '.part must be preserved on exception'
    assert not target.exists()


# Filelock must serialize merges: no lost updates across N threads
def test_concurrent_sessions_no_loss(tmp_path):
    
    target = tmp_path / 'token_usage.txt'
    n = 8

    def worker():
        with token_usage_session(target) as part:
            _write(part, {'together:m1': {'prompt_tokens': 10, 'completion_tokens': 1}})

    threads = [threading.Thread(target=worker) for _ in range(n)]

    for t in threads: t.start()
    for t in threads: t.join()

    result = json.loads(target.read_text(encoding='utf-8'))

    assert result == {'together:m1': {'prompt_tokens': 10 * n, 'completion_tokens': 1 * n}}
    assert list(tmp_path.glob('*.part')) == []
