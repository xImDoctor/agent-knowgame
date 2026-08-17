"""Probe scenario: ask a single informed agent if it wants to share, then exit.

Measures share=true probability in round 1 across varying N, K, P.
There are:
    N - number of agents in the game (n_agents)
    K - number of informed agents (m_informed)
    P - price of sharing (share_cost)

The engine is not started - we hand-build the agent context (system, answer,
faked assistant answer with the token, share question) and make one structured
LLM call per combination. Anonymous mode only.

The prompt is locally patched to drop the reasoning field from the JSON
schema instruction. Thisa probe uses min ShareBoolResponse to save
output tokens. For reasoning-capable models, Together's reasoning field
is extracted separately via TogetherLLMClient.structured_call_with_reasoning.

Grid YAML format:
    n_agents:   [2, 3, 6, 10]
    m_informed: [[1], [1, 2], [1, 3, 5], [1, 3, 5, 7, 9]]   # parallel to n_agents
    share_cost is computed per k = m_informed as unique of [0.1, 1, k/2, k]
    with k/2 included only when strictly between 1 and k.
    seeds:      [42, 43, ..., 51]                           # full list
    early_stopping: true|false                              # default true
    model, api_type, reasoning, request_timeout

Early stopping: when true, the first half of seeds runs as batch 1.
If all results in this batch 1 are identical the config stops there, otherwise
the second half of seeds runs as batch 2. When false, all seeds run without
extra batches or any kind of stopping.

Output:
    probes/probe_share_<sanitized_model>_<YYYY-MM-DD>.jsonl
    probes/probe_share_<sanitized_model>_<YYYY-MM-DD>.csv
"""

import sys
import argparse
import csv
import json
import re
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
from game.states import AgentState, ChatMessage, RoundResult


STUB_TOKEN = 'color'

# match the reasoning-field instruction inside the share prompt and drop it
# leaving only {"share": true|false}
REASONING_STRIP_RE = re.compile(r',\s*"reasoning"\s*:\s*"\.\.\."')


# min schema to validate only true/false field
class ShareBoolResponse(BaseModel):
    share: bool


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

    raise NotImplementedError(f"api_type={config.api_type!r} not supported by probe_share")


# request_timeout added for ollama model variations and it literally depends on hardware
def stub_config(n_agents: int, m_informed: int, share_cost: float, seed: int,
                model: str, api_type: Literal['ollama', 'together', 'fake'],
                request_timeout: float = 60.0) -> GameConfig:
    
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
    )


# drops the reasoning field from JSON instruction in share prompt
# using REASONING_STRIP_RE const from above
def patch_share_prompt(text: str) -> str:
    return REASONING_STRIP_RE.sub('', text)


def make_messages(config: GameConfig) -> list[ChatMessage]:
    """Hand-build the context: system + answer_q + faked assistant + share_q"""
    
    prompts = create_prompt_builder(config, STUB_TOKEN)

    # each player scores correct_count points, so after round 1
    # every player has starting_capital + K (informed all answered correctly)
    agent = AgentState(
        agent_id='agent_0',
        knows_token=True,
        score=config.starting_capital + config.m_informed,
    )

    all_ids = [f'agent_{i}' for i in range(config.n_agents)]

    round_result = RoundResult(
        round_num=1,
        answers={},
        correct_answers={},
        scores_after={},
        correct_count=config.m_informed,
    )

    system_prompt = prompts.build_system_prompt(agent, all_ids)
    answer_prompt = prompts.build_answer_prompt(agent, round_num=1)
    share_prompt = patch_share_prompt(prompts.build_share_prompt(agent, round_result, unknowing_agents=[]))

    # all msg context in the dict format
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': answer_prompt},
        {'role': 'assistant', 'content': json.dumps({'answer': STUB_TOKEN})},
        {'role': 'user', 'content': share_prompt},
    ]


def run_one(config: GameConfig, use_reasoning: bool,
            token_log_path: Path | None = None) -> dict:
    """Runs a single share-probe call. Returns a flat record for JSONL"""

    messages = make_messages(config)

    record = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'model': config.model,
        'api_type': config.api_type,
        'n_agents': config.n_agents,
        'm_informed': config.m_informed,
        'share_cost': config.share_cost,
        'seed': config.seed,
        'share': None,
        'reasoning': None,
        'error': None,
    }

    try:
        llm = build_llm(config, token_log_path=token_log_path)

        # if together and model is marked as reasoning one
        if use_reasoning and isinstance(llm, TogetherLLMClient):
            response, reasoning = llm.structured_call_with_reasoning(messages, ShareBoolResponse)
            record['share'] = response.share
            record['reasoning'] = reasoning
        else:
            response = llm.structured_call(messages, ShareBoolResponse)
            record['share'] = response.share

    except Exception as e:
        record['error'] = f'{type(e).__name__}: {e}'

    return record


def share_costs_for_k(k: int) -> list[float]:
    """
    Makes price logic. Prices per k per spec:
    [0.1, 1, k/2, k] with k/2 included only if 1 < k/2 < k.
    """
    prices = {0.1, 1.0, float(k)}
    half = k / 2

    if 1.0 < half < float(k):
        prices.add(half)

    return sorted(prices)


def expand_grid(grid: dict) -> list[dict]:
    """Parallel lists n_agents x m_informed_lists, cartesian with prices and seeds"""
    
    n_list = grid['n_agents']
    m_lists = grid['m_informed']

    if len(n_list) != len(m_lists):
        raise ValueError(f'n_agents ({len(n_list)}) and m_informed ({len(m_lists)}) must be parallel')

    seeds = list(grid['seeds'])
    early_stopping = bool(grid.get('early_stopping', True))
    model = grid['model']
    api_type = grid['api_type']
    request_timeout = grid.get('request_timeout', 60.0)

    # optional explicit price list, two forms accepted:
    #   share_costs: [0.1, 1, ...]            - flat list, applied to every K in this run
    #   share_costs: {19: [0.1, 1, 4.5, ...]} - per-K dict; K not listed falls back to share_costs_for_k(k)
    # if absent, fall back to share_costs_for_k(k) per the default spec.
    share_costs_override = grid.get('share_costs')

    if early_stopping:  # split into 2 batches, stop if everything in first_seeds the same
        half = len(seeds) // 2
        first_seeds, extra_seeds = seeds[:half], seeds[half:]
    else:
        first_seeds, extra_seeds = seeds, []

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
        # allow both [1, 2] and a bare int like 1
        m_options = m_options if isinstance(m_options, list) else [m_options]

        for k in m_options:

            if not 0 < k < n:
                continue

            for p in _costs_for(k):
                configs.append({
                    'n_agents': n, 'm_informed': k, 'share_cost': p,
                    'model': model, 'api_type': api_type,
                    'request_timeout': request_timeout,
                    'first_seeds': list(first_seeds),
                    'extra_seeds': list(extra_seeds),
                })

    return configs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Probe share probability for informed agents in round 1.')
    p.add_argument('--grid-file', type=str, required=True,
                   help='YAML grid config (see docstring)')
    p.add_argument('--output-dir', type=str, default='probes',
                   help='Where to write jsonl and csv (default: probes/)')
    return p.parse_args()


# for model names and such cases
def sanitize(name: str) -> str:
    return name.replace('/', '-').replace(':', '-').replace(' ', '_')


def default_output_paths(model: str, output_dir: str) -> tuple[Path, Path]:
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    stem = f'probe_share_{sanitize(model)}_{day}'
    base = Path(output_dir)
    return base / f'{stem}.jsonl', base / f'{stem}.csv'


def _write_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Converts the JSONL to a flat CSV when run is ended."""
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        return
    fields = ['ts', 'model', 'api_type', 'n_agents', 'm_informed', 'share_cost',
              'seed', 'share', 'reasoning', 'error']
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

    jsonl_path, csv_path = default_output_paths(model, args.output_dir)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    total_configs = len(configs)
    print(f'Configs to probe: {total_configs}; reasoning={use_reasoning}')
    print(f'Writing to: {jsonl_path}')

    with token_usage_session() as tok, jsonl_path.open('w', encoding='utf-8') as f:
        for ci, cfg_pack in enumerate(configs, 1):
            first_seeds = cfg_pack.pop('first_seeds')
            extra_seeds = cfg_pack.pop('extra_seeds')

            # first batch
            first_shares: list[bool | None] = []
            for seed in first_seeds:
                config = stub_config(**cfg_pack, seed=seed)
                record = run_one(config, use_reasoning, token_log_path=tok)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                first_shares.append(record['share'])
                print(f'  [{ci}/{total_configs}] '
                      f'n={record["n_agents"]:>2} k={record["m_informed"]:>2} '
                      f'p={record["share_cost"]:<5} seed={seed:>3} '
                      f'share={record["share"]} err={record["error"]}')

            # early stopping: if all first-batch shares are identical (and no errors) than
            # skip extra seeds
            # otherwise run the extra batch
            unique = {s for s in first_shares if s is not None}
            if len(unique) <= 1:
                continue

            for seed in extra_seeds:
                config = stub_config(**cfg_pack, seed=seed)
                record = run_one(config, use_reasoning, token_log_path=tok)

                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()

                print(f'  [{ci}/{total_configs} extra] '
                      f'n={record["n_agents"]:>2} k={record["m_informed"]:>2} '
                      f'p={record["share_cost"]:<5} seed={seed:>3} '
                      f'share={record["share"]} err={record["error"]}')

    _write_csv(jsonl_path, csv_path)

    print(f'\nJSONL: {jsonl_path}')
    print(f'CSV:   {csv_path}')


if __name__ == '__main__':
    main()
