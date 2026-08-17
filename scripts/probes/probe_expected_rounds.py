"""Probe scenario: ask a single informed agent how many rounds it expects the
game to last, doing it right after the system prompt. 
No answer phase or faked reply.

Measures the integer expectation across varying N, K, C for one informed
agent, in anonymous mode. Where:
    N - number of agents in the game (n_agents)
    K - number of informed agents (m_informed)
    C - cost/price of sharing (share_cost)

Same grid loader as scripts/probe_share.py, but
without early stopping (integer answers are almost never all-identical
across seeds so early stopping is pointless in this case).

For reasoning-capable Together models the reasoning trace is captured via
TogetherLLMClient.structured_call_with_reasoning.

Grid YAML format (shared with probe_share.py):
    n_agents:   [2, 3, 6, 10]
    m_informed: [[1], [1, 2], [1, 3, 5], [1, 3, 5, 7, 9]]
    seeds:      [42, 43, ..., 51]
    model, api_type, reasoning, request_timeout

In addition, included optional field 'payment_mode'.
YAML config:
    payment_mode: 'teacher_pays' | 'student_pays'
If it is skipped, uses 'teacher_pays' by default (standard run)
    
early_stopping is ignored here.

Output:
    probes/probe_expected_rounds_<sanitized_model>_<YYYY-MM-DD_HH-MM-SS>_<N>seeds.jsonl
    probes/probe_expected_rounds_<sanitized_model>_<YYYY-MM-DD_HH-MM-SS>_<N>seeds.csv
"""

import sys
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from game.config import GameConfig
from game.clients import LLMClient, OllamaLLMClient, TogetherLLMClient
from game.clients.token_usage import token_usage_session
from game.prompt_builder import create_prompt_builder
from game.states import AgentState, ChatMessage


STUB_TOKEN = 'color'

USER_QUESTION_BASELINE = (
    'Knowing everything above, what is your expectation concerning the game '
    'length in rounds? Answer as JSON {"number": int}'
)

USER_QUESTION_V1 = (
    'Knowing everything above, give your best guess concerning the game '
    'length in rounds? Answer as JSON {"number": int}'
)

user_question = USER_QUESTION_V1

# min schema to validate only integer expectation
class RoundsExpectation(BaseModel):
    number: int


def build_llm(config: GameConfig, token_log_path: Path | None = None) -> LLMClient:
    log_path = token_log_path if token_log_path is not None else Path('token_usage.txt')

    if config.api_type == 'ollama':
        return OllamaLLMClient(
            model=config.model,
            seed=config.seed,
            temperature=config.temperature,
            top_p=config.top_p,
            request_timeout=config.request_timeout,
            token_log_path=log_path,
        )

    if config.api_type == 'together':
        return TogetherLLMClient(
            model=config.model,
            seed=config.seed,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            request_timeout=config.request_timeout,
            token_log_path=log_path,
        )

    raise NotImplementedError(f"api_type={config.api_type!r} not supported by probe_expected_rounds")


def stub_config(n_agents: int,
                m_informed: int,
                share_cost: float,
                seed: int,
                model: str,
                api_type: Literal['ollama', 'together', 'fake'],
                request_timeout: float = 60.0,
                payment_mode: Literal['teacher_pays', 'student_pays', 'split'] = 'teacher_pays',
) -> GameConfig:

    initiation_mode = 'student_only' if payment_mode == 'student_pays' else 'teacher_only'

    return GameConfig(
        n_agents=n_agents,
        m_informed=m_informed,
        share_cost=share_cost,
        starting_capital=0.0,
        max_rounds=1,
        seed=seed,
        model=model,
        api_type=api_type,
        request_timeout=request_timeout,
        template_version='v1_baseline',
        matcher='first_come',
        initiation_mode=initiation_mode,
        payment_mode=payment_mode,
    )


def make_messages(config: GameConfig) -> list[ChatMessage]:
    """Hand-build the context with system prompt + the expectation question only"""

    prompts = create_prompt_builder(config, STUB_TOKEN)

    # informed or uninformed agent that has not played yet
    # score is at starting_capital
    agent = AgentState(
        agent_id='agent_0',
        knows_token=(config.payment_mode == 'teacher_pays'), # true if informed (teacher pays)
        score=config.starting_capital,
    )

    all_ids = [f'agent_{i}' for i in range(config.n_agents)]

    system_prompt = prompts.build_system_prompt(agent, all_ids)

    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_question},
    ]


def run_one(config: GameConfig, use_reasoning: bool,
            token_log_path: Path | None = None) -> dict:
    """Single expectation probe call. Returns a flat record for JSONL"""

    messages = make_messages(config)

    record = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'model': config.model,
        'api_type': config.api_type,
        'n_agents': config.n_agents,
        'm_informed': config.m_informed,
        'share_cost': config.share_cost,
        'seed': config.seed,
        'number': None,
        'reasoning': None,
        'error': None,
    }

    try:
        llm = build_llm(config, token_log_path=token_log_path)

        if use_reasoning and isinstance(llm, TogetherLLMClient): # if turned on reasoning field for such model
            response, reasoning = llm.structured_call_with_reasoning(messages, RoundsExpectation)
            record['number'] = response.number
            record['reasoning'] = reasoning
        else:
            response = llm.structured_call(messages, RoundsExpectation)
            record['number'] = response.number

    except Exception as e:
        record['error'] = f'{type(e).__name__}: {e}'

    return record


def share_costs_for_k(k: int) -> list[float]:
    """Prices per k per spec: [0.1, 1, k/2, k] with k/2 included only if 1 < k/2 < k"""
    prices = {0.1, 1.0, float(k)}
    half = k / 2
    if 1.0 < half < float(k):
        prices.add(half)
    return sorted(prices)


def expand_grid(grid: dict) -> list[dict]:
    """Same layout as probe_share.expand_grid, but early_stopping is ignored."""

    n_list = grid['n_agents']
    m_lists = grid['m_informed']

    if len(n_list) != len(m_lists):
        raise ValueError(f'n_agents ({len(n_list)}) and m_informed ({len(m_lists)}) must be parallel')

    seeds = list(grid['seeds'])
    model = grid['model']
    api_type = grid['api_type']
    request_timeout = grid.get('request_timeout', 60.0)

    # if used student_pays get else - default teacher_pays
    payment_mode = grid.get('payment_mode', 'teacher_pays')

    # optional explicit price list, two forms accepted:
    #   share_costs: [0.1, 1, ...]            - flat list, applied to every K in this run
    #   share_costs: {19: [0.1, 1, 4.5, ...]} - per-K dict; K not listed falls back to share_costs_for_k(k)
    # if absent, fall back to share_costs_for_k(k) per the default spec.
    share_costs_override = grid.get('share_costs')

    def _costs_for(k: int) -> list[float]:

        if isinstance(share_costs_override, list):
            return list(share_costs_override)
        
        if isinstance(share_costs_override, dict):
            per_k = share_costs_override.get(k) or share_costs_override.get(str(k))
            if per_k is not None:
                return list(per_k)
            
        return share_costs_for_k(k)

    configs: list[dict] = []
    for n, m_options in zip(n_list, m_lists):
        m_options = m_options if isinstance(m_options, list) else [m_options]

        for k in m_options:
            if not 0 < k < n:
                continue

            for p in _costs_for(k):
                configs.append({
                    'n_agents': n, 'm_informed': k, 'share_cost': p,
                    'model': model, 'api_type': api_type,
                    'request_timeout': request_timeout,
                    'seeds': list(seeds),
                    'payment_mode': payment_mode,
                })

    return configs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Probe how many rounds an informed agent expects.')
    p.add_argument('--grid-file', type=str, required=True,
                   help='YAML grid config (shared with probe_share.py)')
    p.add_argument('--output-dir', type=str, default='probes',
                   help='Where to write jsonl and csv (default: probes/)')
    
    return p.parse_args()


def sanitize(name: str) -> str:
    return name.replace('/', '-').replace(':', '-').replace(' ', '_')


def default_output_paths(model: str, output_dir: str, n_seeds: int, payment_mode: str) -> tuple[Path, Path]:
    # include the start timestamp and seed count so parallel prompt-variant
    # runs do not overwrite each other
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    stem = f'probe_expected_rounds_{sanitize(model)}_{ts}_{n_seeds}seeds_{sanitize(payment_mode)}'
    base = Path(output_dir)

    return base / f'{stem}.jsonl', base / f'{stem}.csv'


def _write_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    
    if not rows:
        return
    
    fields = ['ts', 'model', 'api_type', 'n_agents', 'm_informed', 'share_cost',
              'seed', 'number', 'reasoning', 'error']
    
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    load_dotenv()
    args = parse_args()

    grid = yaml.safe_load(Path(args.grid_file).read_text(encoding='utf-8'))
    configs = expand_grid(grid)
    use_reasoning = bool(grid.get('reasoning', False))
    model = grid['model']
    n_seeds = len(grid['seeds'])
    payment_mode = str(grid.get('payment_mode', 'teacher_pays'))

    jsonl_path, csv_path = default_output_paths(model, args.output_dir, n_seeds, payment_mode)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    total_configs = len(configs)
    print(f'Configs to probe: {total_configs}; reasoning={use_reasoning}')
    print(f'Writing to: {jsonl_path}')

    with token_usage_session() as tok, jsonl_path.open('w', encoding='utf-8') as f:
        for ci, cfg_pack in enumerate(configs, 1):
            seeds = cfg_pack.pop('seeds')

            for seed in seeds:
                config = stub_config(**cfg_pack, seed=seed)
                record = run_one(config, use_reasoning, token_log_path=tok)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                print(f'  [{ci}/{total_configs}] '
                      f'n={record["n_agents"]:>2} k={record["m_informed"]:>2} '
                      f'p={record["share_cost"]:<5} seed={seed:>3} '
                      f'number={record["number"]} err={record["error"]}')

    _write_csv(jsonl_path, csv_path)

    print(f'\nJSONL: {jsonl_path}')
    print(f'CSV:   {csv_path}')


if __name__ == '__main__':
    main()
