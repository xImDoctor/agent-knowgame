"""Dump all prompt templates of a given builder version to markdown files.

Script for prompt extraction to preview and edit without
running the engine.
Stubs replace runtime values (token, agent ids, scores).

Two files are written per run - one per game mode:
  prompts_<version>_anonymous.md  (display_names_file unset)
  prompts_<version>_named.md      (display_names_file set; skipped if missing)

Usage:
    python scripts/utils/dump_prompts.py
    python scripts/utils/dump_prompts.py --template-version v1_baseline
    python scripts/utils/dump_prompts.py --display-names-file configs/names_test_pool.txt
"""

import sys
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from game.config import GameConfig
from game.prompt_builder import create_prompt_builder
from game.states import GameState, RoundResult


STUB_TOKEN = '<TOKEN>'
DEFAULT_NAMES_FILE = Path('configs/names_test_pool.txt')


def stub_config(template_version: str, display_names_file: Path | None = None) -> GameConfig:
    # min config that passes validation, values are placeholders
    return GameConfig(
        n_agents=4,
        m_informed=1,
        share_cost=0.01,
        starting_capital=1.0,
        max_rounds=5,
        seed=42,
        model='<MODEL>',
        api_type='fake',
        template_version=template_version,
        display_names_file=display_names_file,
    )


def stub_round_result() -> RoundResult:
    return RoundResult(
        round_num=1,
        answers={
            'agent_0': '<TOKEN>',
            'agent_1': '<unknown>',
            'agent_2': '<TOKEN>',
            'agent_3': '<unknown>',
        },
        correct_answers={
            'agent_0': True,
            'agent_1': False,
            'agent_2': True,
            'agent_3': False,
        },
        scores_after={
            'agent_0': 1.5,
            'agent_1': 1.5,
            'agent_2': 1.5,
            'agent_3': 1.5,
        },
        correct_count=2,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Dump all prompt templates of a builder to markdown.')

    p.add_argument('--template-version', type=str, default='v1_baseline')
    p.add_argument('--display-names-file', type=str, default=str(DEFAULT_NAMES_FILE),
                   help='Path to display-names txt for the named dump (default: configs/names_test_pool.txt)')
    p.add_argument('--output-dir', type=str, default='.',
                   help='Where to write prompts_*.md files (default: repo root)')

    return p.parse_args()


def dump_prompts(config: GameConfig, output_path: Path) -> None:

    builder = create_prompt_builder(config, STUB_TOKEN)

    # use real initialize to resolve display_names consistently with engine runtime
    state = GameState.initialize(config, random.Random(config.seed))

    # force predictable roles where agent_0 knows, agent_1 is unknowing
    knowing = state.agents[0]
    knowing.knows_token = True
    unknowing = state.agents[1]
    unknowing.knows_token = False

    all_names = [a.display_name for a in state.agents]
    id_to_display = {a.agent_id: a.display_name for a in state.agents}
    round_result = stub_round_result()
    unknowing_names = [state.agents[1].display_name, state.agents[3].display_name]

    sections = [
        ('build_system_prompt (knowing agent)',
            builder.build_system_prompt(knowing, all_names)),
        ('build_system_prompt (unknowing agent)',
            builder.build_system_prompt(unknowing, all_names)),
        ('build_answer_prompt',
            builder.build_answer_prompt(knowing, round_num=1)),
        ('build_share_prompt',
            builder.build_share_prompt(knowing, round_result, unknowing_names)),
    ]

    # retry prompt is only used in named mode, anon ver skips communication retry
    if not config.is_anonymous:
        sections.append((
            'build_retry_share_prompt',
            builder.build_retry_share_prompt(
                knowing, round_result, [unknowing_names[-1]], previous_target=unknowing_names[0]),
        ))

    sections.extend([
        ('build_round_summary',
            builder.build_round_summary(knowing, round_result, id_to_display)),
        ('build_transfer_token_prompt',
            builder.build_transfer_token_prompt(state.agents[0].display_name)),
    ])

    mode = 'anonymous' if config.is_anonymous else 'named'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'# Prompts dump - {config.template_version} ({mode})\n\n')
        f.write(f'Stubs: token=`<TOKEN>`, n_agents={config.n_agents}, scores=placeholder.\n')
        if not config.is_anonymous:
            f.write(f'Names from: `{config.display_names_file}`\n')
        f.write('\n')
        for title, text in sections:
            f.write(f'## {_md_section_title(title)}\n\n```\n{text}\n```\n\n')


def _md_section_title(title: str) -> str:
    # wrap first word (method name with underscores) in backticks
    # so md does not interpret '_' as emphasis
    head, _, tail = title.partition(' ')
    return f'`{head}` {tail}'.rstrip()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # anon mode dump - written anyway/always
    anon_cfg = stub_config(args.template_version, display_names_file=None)
    anon_path = out_dir / f'prompts_{args.template_version}_anonymous.md'

    dump_prompts(anon_cfg, anon_path)
    print(f'Wrote {anon_path}')

    # named mode dump - only if the names file exists
    names_file = Path(args.display_names_file)
    if not names_file.exists():
        print(f'Skipping named dump - {names_file} not found')
        return

    named_cfg = stub_config(args.template_version, display_names_file=names_file)
    named_path = out_dir / f'prompts_{args.template_version}_named.md'

    dump_prompts(named_cfg, named_path)
    print(f'Wrote {named_path}')


if __name__ == '__main__':
    main()
