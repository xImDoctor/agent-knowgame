# Propagate Bench

A research project (game) for studying **free-riding in knowledge transfer between LLM agents**.

A configurable multi-agent game engine, plus one-call probe scripts that let you measure willingness to share, expected game length and other behavioural effects across a grid of configurations and models.

---

## Table of contents

1. [Overview](#overview)
2. [How the game works](#how-the-game-works)
   - [Round phases](#round-phases)
   - [Agent scoring](#scoring)
   - [Game stop conditions](#stop-conditions)
   - [Prompt building and variants](#prompt-variants)
   - [Matcher logic](#matcher-logic)
3. [Install](#install)
4. [Running a full game](#running-a-full-game)
   - [CLI arguments](#cli-arguments)
   - [YAML config](#yaml-config)
   - [Config fields reference](#config-fields-reference)
5. [Experimental probes](#experimental-probes)
   - [Grid probes: shared grid YAML format](#grid-probes-shared-grid-yaml-format)
   - [`probe_share.py` – will the informed agent share?](#probe_sharepy--will-the-informed-agent-share)
   - [`probe_request.py` – will the uninformed agent request the word from the game?](#probe_requestpy--will-the-uninformed-agent-request-the-word-from-the-game)
   - [`probe_expected_rounds.py` – how long does the agent think the game will last?](#probe_expected_roundspy--how-long-does-the-agent-think-the-game-will-last)
   - [`probe_together.py` – Together API sanity check](#probe_togetherpy--together-api-sanity-check)
6. [Utility scripts](#utility-scripts)
7. [Cost accounting](#cost-accounting)
8. [Tests](#tests)
9. [Repository layout](#repository-layout)

---

## Overview

The game puts `N` LLM agents in a round-based setting where `K < N` of them start with a secret **token** or **word** (a random 4-digit string or a string entered by the user). Every round each agent is asked for the token, and correct answers earn points **for the whole group**, giving `+1` point to each agent for every correct answer that round. Agents that already know the token may (or may not) choose to teach an unknowing agent, paying `share_cost` (`C`) for the privilege – this is where free-riding shows up.

The engine supports full multi-round games (`scripts/run_game.py`), but most current experiments in this repo rely on the **probes** – single-call scripts that skip the engine (for the sake of simplification) and measure one specific quantity (share decision, expected rounds, etc.) on a grid of `(N, K, C, seed)` configurations.

---

## How the game works

### Round phases

Every round `r` runs three phases, in order:

1. **Answer phase.** Every agent is asked for the token. The answer schema is `AnswerResponse = {"answer": str}`. Whether the answer equals the token is recorded.
2. **Scoring phase.** All agents (informed or not) receive `+correct_count` points for that round. See [Scoring](#scoring).
3. **Communication phase.** Every informed agent decides whether to teach one specific unknowing agent this round. If it decides to share, it pays `share_cost` and the chosen student receives the token. Simultaneous choices onto the same student are resolved by the [matcher](#matcher-logic).

Each event of the round is written to a JSONL log under `logs/`.


### Scoring

Each round every agent (informed or not) receives `+correct_count` points – i.e. the total number of agents that answered correctly that round. 

When an informed agent teaches, it pays `share_cost` immediately. In `teacher_pays` mode (the only implemented one now) the student is free.

### Stop conditions

The game ends when any of these conditions becomes true:

- **`all_know`** – every agent has learned the token.
- **`max_rounds`** – the round counter hit `max_rounds`.
- **`format_limit_exhausted`** – an agent failed schema validation more than `max_retries` times in a row on the same phase (or another error led to more than `max_retries` attempts). The game stops cleanly with a stuck-info payload in the `game_over` event.

### Prompt variants

Prompt templates live in `game/prompt_builder.py`. The active version is `v1_baseline`, with two flavours:

- **Anonymous** (`display_names_file is None`) – agents are addressed by opaque IDs like `agent_0`. At the same time, the agents themselves do not see each other's names and cannot use them. When matching, the first available student is taken.
- **Named** – agents receive human-looking display names loaded from a file and can use them for student matching while transferring the token.

Dump the exact rendered prompts with:

```bash
python scripts/utils/dump_prompts.py --template-version v1_baseline
```

This writes `prompts_v1_baseline_anonymous.md` and `prompts_v1_baseline_named.md`.

> [!WARNING]
> The template requested by the script must exist in `PROMPT_BUILDER_REGISTRY` to be dumped.

### Matcher logic

When the game is launched anonymously, and the agents do not know each other's names, matching occurs without displaying the names and interacting with them. To transfer the token at the request of a knowledgeable agent, the first unknowing agent (student) in the list is selected.

For a game where agents have display names and choose the student to whom the token will be given by name, two matching logics are implemented. They differ in how they resolve the situation when two knowing agents pick the same student. The `matcher` decides who wins the tie:

- **`random_choice`** – random tie-break; the loser's transfer is dropped.
- **`first_come`** – the first-declared transfer wins; the loser gets feedback ("that student is already busy") and may pick another student or decline.

In a lose case, the declined knowing agent gets an option to choose another student or not share at all.

---

## Install

To start working with this repo you need to create a virtual environment and install the Python dependencies:

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
```

Backends:

- **`fake`** – no external dependency; uses `FakeLLMClient` with a scripted `always_share` / `never_share` / `random_share` strategy. This is not a full-fledged game mode (there are no LLM agents); it is used to test the engine, the logger and the automated tests.
- **`ollama`** – needs a locally running `ollama serve`.
- **`together`** – needs `TOGETHER_API_KEY` in a `.env` file at the repo root.

---

## Running a full game

The main entry point is `scripts/run_game.py`. It accepts a YAML config, inline CLI flags, or both (note that CLI overrides YAML).

```bash
# YAML
python scripts/run_game.py configs/config_template.yaml

# fully inline (no YAML)
python scripts/run_game.py --n-agents 3 --m-informed 1 --share-cost 1 \
    --max-rounds 2 --model fake --api-type fake

# YAML + partial inline override
python scripts/run_game.py configs/config_template.yaml --seed 7
python scripts/run_game.py configs/config_template.yaml --max-rounds 5 --model llama3.2:3b
```

Console inline args overwrite values from the YAML. Log traces (JSONL) land under `logs/`.

### CLI arguments

| Flag | Type | Meaning |
|---|---|---|
| `config` (positional) | path | path to a YAML config; optional if everything is passed inline |
| `--n-agents` | int | total number of agents (`>= 2`) |
| `--m-informed` | int | agents knowing token from start (`0 < m < n`) |
| `--share-cost` | float | cost of token transfer (teaching) |
| `--starting-capital` | float | initial score of every agent (default `0.0`) |
| `--max-rounds` | int | upper bound on number of game rounds |
| `--seed` | int | RNG seed: token, informed pick, tie-break |
| `--model` | str | model name (e.g. `llama3.2:3b`, `openai/gpt-oss-20b`, `fake`) |
| `--api-type` | str | `fake` / `ollama` / `together` |
| `--temperature` | float | LLM sampling temperature |
| `--top-p` | float | LLM sampling nucleus |
| `--request-timeout` | float | seconds per LLM call (default `60`) |
| `--template-version` | str | key in `PROMPT_BUILDER_REGISTRY` (`v1_baseline` now) |
| `--matcher` | str | `random_choice` / `first_come` |
| `--fake-strategy` | str | `FakeLLMClient` share-phase strategy: `always_share` / `never_share` / `random_share` |

> [!IMPORTANT]
> The reserved fields (`initiation_mode`, `payment_mode`, `token_transfer_mode`) previously worked only with default values (`teacher_only`, `teacher_pays`, `direct`). Valid (available) pairs are now provided `(teacher_only, teacher_pays)`, `(student_only, student_pays)`, the token transfer mode is still `direct` only.

> [!WARNING]
> If you try to select modes that are not implemented, the game will not start (fallback with a warning).

### YAML config

`configs/config_template.yaml` is the base working template – copy and tweak per experiment. Key names match `GameConfig` fields. Minimum:

```yaml
n_agents: 5
m_informed: 2
share_cost: 1.0
max_rounds: 20
seed: 42

model: llama3.2:3b
api_type: ollama
temperature: 0.7
top_p: 1.0

template_version: v1_baseline
matcher: first_come
```

To run without any LLM, use `model: fake`, `api_type: fake`. The default fake strategy is `always_share`; swap it via the `--fake-strategy` CLI argument.

### Config fields reference

Full list of fields validated by `GameConfig` (`game/config.py`):

| Field | Default | Notes |
|---|---|---|
| `n_agents` | required | `>= 2` |
| `m_informed` | required | `0 < m < n_agents` |
| `share_cost` | required | `>= 0` |
| `starting_capital` | `0.0` | seed score for every agent |
| `token` | `None` | fixed token; if `None`, a random 4-digit string is generated |
| `seed` | `42` | RNG seed |
| `max_rounds` | `1` | `>= 1` |
| `display_names_file` | `None` | opt-in 'named' mode |
| `display_names_random` | `False` | randomised name pick (deterministic per seed) |
| `model` | required | model id |
| `api_type` | required | `ollama` / `together` / `fake` |
| `top_p`, `temperature` | `1.0`, `0.7` | LLM sampling |
| `request_timeout` | `60.0` | seconds per LLM call |
| `max_retries` | `1` | schema retries per call before stopping the game |
| `max_tokens` | `4112` | API max tokens |
| `template_version` | `v1_baseline` | prompt template key |
| `matcher` | `first_come` | tie resolution |
| `initiation_mode`, `payment_mode`, `token_transfer_mode` | `teacher_only`, `teacher_pays`, `direct` | reserved; only defaults implemented now |

---

## Experimental probes

Probes are one-call scripts that bypass the game engine to measure a single behavioural signal on a grid of `(N, K, C, seed)` configurations, where `N` is the number of agents in the game, `K` is the number of agents that know the token, `C` is the cost of sharing the token, and `seed` affects the LLM output with the other settings fixed (different `seeds` can be interpreted as independent runs with everything else held constant).

The probes hand-build the agent context (system prompt + optionally an answer-phase reply) and make **one** structured LLM call per grid cell. They are the workhorses of the current experiments.

Multiple probes can be launched in parallel — each script isolates its token counter into a per-process `.part` file merged into the canonical `token_usage.txt` atomically on exit. See [Cost accounting](#cost-accounting) for the mechanism and how to recover parts from crashed runs.


### Grid probes: shared grid YAML format

`probe_share.py`, `probe_request.py` and `probe_expected_rounds.py` share the same YAML:

```yaml
n_agents:   [2, 3, 6, 10]                                # population sizes N
m_informed: [[1], [1, 2], [1, 3, 5], [1, 3, 5, 7, 9]]    # per-N K values (parallel to n_agents)
seeds:      [42, 43, ..., 51]                            # RNG seed pool (number of different runs with other parameters fixed)
early_stopping: true                                     # split seeds in half; skip 2nd if 1st is uniform
model:      openai/gpt-oss-20b                           # LLM acting as agents
api_type:   together                                     # API provider (could be together, ollama or fake)
reasoning:  true                                         # request reasoning trace (Together only)
request_timeout: 240

# optional — override the default per-K price grid:
#   flat list  → same set of C values applied to every K
#   per-K dict → K not listed falls back to the default share_costs_for_k(k)
share_costs:
  19: [0.1, 1, 4.5, 9.5, 14, 19]

# optional — only for probe_expected_rounds:
#   'teacher_pays' (default) — asks an informed agent, system prompt mentions the sender pays a fee
#   'student_pays'           — asks an uninformed agent, system prompt mentions the requester pays
payment_mode: teacher_pays
```

`share_cost` per `K` is derived automatically as unique values of `[0.1, 1, K/2, K]` (`K/2` included only if `1 < K/2 < K`). **Total: 35 cells** for the standard grid; extra cells appear per any `share_costs` override. `probe_share.py` and `probe_request.py` ignore `payment_mode` (each script fixes its own mode); only `probe_expected_rounds.py` honours it (because works with both modes)

Ready-made experiment configs in [`configs/probes/`](/configs/probes):

| File | Purpose |
|---|---|
| `probe_ollama_smoke.yaml` | tiny smoke on local Ollama, `early_stopping: true` |
| `probe_<model>_10seeds.yaml` | 10-seed grid on Together API, `early_stopping: true` |
| `probe_<model>_100seeds.yaml` | 100-seed grid on Together API, `early_stopping: false` |
| `probe_<model>_N20_K19_100seeds.yaml` | "all but one know" slice at N=20 with a custom `share_costs` sweep |

### `probe_share.py` – will the informed agent share?

Measures **`p(share=True)`** in the first round, per `(N, K, C, seed)`. Uses a minimal `ShareBoolResponse = {"share": bool}` schema to save output tokens; the prompt template's `"reasoning"` clause is locally regex-patched out. For reasoning-capable Together models the reasoning trace is captured separately.

Fixes the game to `teacher_only` initiation + `teacher_pays` payment (fixes these game modes). The stub token used in the faked answer phase is `STUB_TOKEN = 'color'`. Change the module-level constant if you need to swap the word.

> [!WARNING]
> Currently works in **anonymous** mode only.

Run:

```bash
python scripts/probes/probe_share.py --grid-file configs/probes/probe_qwen3_235b_10seeds.yaml
```

**Early stopping.** When enabled, the first half of `seeds` runs as batch 1. If every batch-1 result is identical, the config stops there and does not run batch 2. When disabled, all seeds always run.

**Output:**
- `probes/probe_share_<sanitized_model>_<YYYY-MM-DD>.jsonl`
- `probes/probe_share_<sanitized_model>_<YYYY-MM-DD>.csv`

### `probe_request.py` – will the uninformed agent request the word from the game?

Mirror of `probe_share.py` for the `student_pays` regime. Measures **`p(request=True)`** in the first round, per `(N, K, C, seed)`, from the perspective of an **uninformed** agent that pays a fee to receive the word, and it gives the word **directly from the game** (no specific teacher). Uses schema `RequestResponse = {"request": bool}`.

Fixes the game to `initiation_mode='student_only'` + `payment_mode='student_pays'`. Context construction differs from `probe_share.py`:
- the probed agent has `knows_token=False`
- the faked answer-phase reply is `{"answer": "<unknown>"}` (a student cannot guess, the reply fixes that agent does not know the word)
- the second-stage prompt is `build_request_prompt(...)` (game-level request wording)

> [!WARNING]
> Currently works in **anonymous** mode only.

Run:

```bash
python scripts/probes/probe_request.py --grid-file configs/probes/probe_qwen2_5_7b_turbo_100seeds.yaml
```

**Output:**
- `probes/probe_request_<sanitized_model>_<YYYY-MM-DD>.jsonl`
- `probes/probe_request_<sanitized_model>_<YYYY-MM-DD>.csv`

Column names in the JSONL/CSV: `request` (bool) instead of `share`. All other fields (`n_agents`, `m_informed`, `share_cost`, `seed`, `reasoning`, `error`) match `probe_share.py`.

### `probe_expected_rounds.py` – how long does the agent think the game will last?

Asks a single agent, right after the system prompt (no answer phase, no faked reply), how many rounds it expects the game to last. Response schema is `RoundsExpectation = {"number": int}`.

The follow-up user question in the current `V1` variant is:

> Knowing everything above, give your best guess concerning the game length in rounds? Answer as JSON `{"number": int}`

Earlier `BASELINE` wording used *"what is your expectation concerning..."* and produced a noticeably different distribution. Both variants are kept at the top of the script; switch by reassigning the module-level `user_question`.

**Payment mode.** Optional `payment_mode` field in the YAML grid switches between two regimes:
- `teacher_pays` (default) — probes an **informed** agent; system prompt describes the teacher paying a fee.
- `student_pays` — probes an **uninformed** agent; system prompt describes the requester paying a fee.

Both regimes ask the same follow-up question, so any distributional shift is attributable to the system-prompt framing (or agent role), not to the question itself.

Run:

```bash
python scripts/probes/probe_expected_rounds.py --grid-file configs/probes/probe_qwen3_235b_10seeds.yaml
```

`early_stopping` is **ignored** here: integer answers are almost never identical across seeds, so early stopping would never trigger.

**Output:**
- `probes/probe_expected_rounds_<sanitized_model>_<YYYY-MM-DD_HH-MM-SS>_<N>seeds_<payment_mode>.jsonl`
- `probes/probe_expected_rounds_<sanitized_model>_<YYYY-MM-DD_HH-MM-SS>_<N>seeds_<payment_mode>.csv`

`payment_mode` is embedded in the filename so parallel runs of the same model under different modes do not overwrite each other.

### `probe_together.py` – Together API sanity check

Single-call smoke test against Together. Verifies that `TOGETHER_API_KEY` works, the model is serverless, and Together's grammar engine accepts our pydantic schema (some model backends reject certain schemas).

```bash
python scripts/probes/probe_together.py                            # default: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
python scripts/probes/probe_together.py openai/gpt-oss-20b         # any Together model id
```

> In other words, this is the check that the API is ready for a full game run or for launching any of the other scripts.

---

## Utility scripts

```bash
# dump prompt templates of a builder to markdown - one file per game mode
# writes prompts_<version>_anonymous.md and prompts_<version>_named.md
python scripts/utils/dump_prompts.py --template-version v1_baseline

# estimate API spend from token_usage.txt against the prices table
python scripts/utils/calc_costs.py
```

---

## Cost accounting

Every LLM client writes per-model token counts into `token_usage.txt` at the repo root (append-and-update). Estimate spend at any time:

```bash
python scripts/utils/calc_costs.py
```

The script reads `token_usage.txt` and multiplies each model's prompt/completion counts by the per-1K prices in `configs/together_prices.yaml`. Models without a matching price entry (`fake:*`, `ollama:*`, or newly-introduced Together models) are skipped with a `[WARN] No prices for ...` line.

Update `configs/together_prices.yaml` manually when Together changes its pricing or when you expand the list of models used in a game.

### Parallel-safe accounting

Scripts that open a `token_usage_session()` (`probe_share.py`, `probe_request.py`, `probe_expected_rounds.py`) write to a per-process buffer at the repo root:

    token_usage.<pid>.<timestamp>-<uid>.part

so several scripts can run in parallel without racing on the shared counter. On clean exit the part is merged into `token_usage.txt` atomically under a file lock (`token_usage.txt.lock`) and then deleted.

If a script crashes or is stopped with Ctrl+C its `.part` file is **preserved** for manual recovery. Merge orphan parts later with:

```bash
python scripts/utils/merge_token_usage.py             # merge all orphan parts
python scripts/utils/merge_token_usage.py --dry-run   # list what would be merged
python scripts/utils/merge_token_usage.py --target other_target.txt
```

---

## Tests

```bash
python -m pytest
```

**60 automatic tests** currently, organised by module. What each file covers:

| File | Focus | Highlights |
|---|---|---|
| `tests/test_config.py` | `GameConfig` validation | valid config, frozen (immutable), `n_agents >= 2`, `0 < m_informed < n_agents`, `share_cost >= 0`, reserved-mode NIE, extra-field rejection, `from_yaml` loader |
| `tests/test_states.py` | `GameState` / `AgentState` semantics | deterministic init, `distribute_score` = `+correct_count` per agent (V1Baseline spec), `apply_transfer` teacher/student cost bookkeeping, stop conditions (`all_know`, `max_rounds`), display-name resolution (first-N vs. random-with-seed) |
| `tests/test_fake_llm.py` | `FakeLLMClient` behaviour | knowing agents answer the token, unknowing agents do not reveal it, `always_share` / `never_share` strategies |
| `tests/test_llm_runner.py` | `call_with_retry` schema retries | first-attempt success, retry pops the invalid user turn and keeps the context clean, `FormatLimitExhausted` after exhausting retries, transport errors return `None` (no raise), `format_warning` / `format_limit` events logged, `max_retries=0` means one attempt, `extra_request_payload` merged |
| `tests/test_logger.py` | JSONL event logger | `event` field present, envelope fields (`ts`, `round`, `agent_id`), `set_round` reflected, output is valid JSONL |
| `tests/test_engine_integration.py` | End-to-end fake runs | full fake run terminates at `max_rounds`, `format_limit_exhausted` stops the game cleanly, matcher choice does not affect result in anonymous mode, anonymous share prompt does not leak target list |
| `tests/test_token_usage.py` | Cross-process merge semantics | source-sum into empty/existing target, session merges on clean exit, `.part` preserved on exception, no loss under N concurrent sessions |

Run a single group with, e.g. `python -m pytest tests/test_engine_integration.py -v`.

---

## Repository layout

Overview of the current project working folder:

```
agent-knowgame/
├── game/                    # core game engine
│   ├── config.py            # GameConfig (pydantic, frozen)
│   ├── states.py            # AgentState / GameState / RoundResult
│   ├── engine.py            # GameEngine.run() and phase helpers
│   ├── prompt_builder.py    # v1_baseline prompt templates + response schemas
│   ├── agent_matching.py    # random_choice / first_come matchers
│   ├── llm_runner.py        # call_with_retry, FormatLimitExhausted
│   ├── logger.py            # JSONL EventLogger
│   └── clients/
│       ├── base_client.py   # LLMClient abstract + token accounting
│       ├── fake_llm.py      # scripted FakeLLMClient
│       ├── ollama_llm.py    # OllamaLLMClient
│       ├── together_llm.py  # TogetherLLMClient (+ reasoning capture)
│       └── token_usage.py   # per-process buffer + cross-process merge
├── scripts/
│   ├── run_game.py                # main entry point for full games
│   ├── probes/
│   │   ├── probe_share.py             # p(share) grid probe
│   │   ├── probe_request.py           # p(request) mirror of probe_share.py
│   │   ├── probe_expected_rounds.py   # expected-rounds grid probe (V1)
│   │   └── probe_together.py          # Together API sanity check
│   └── utils/
│       ├── dump_prompts.py            # render templates to markdown
│       ├── calc_costs.py              # spend estimator
│       └── merge_token_usage.py       # merge orphan .part files after crashes
├── configs/
│   ├── config_template.yaml       # base full-game config
│   ├── together_prices.yaml       # per-1K price table
│   └── probes/                    # grid configs for probe scripts
├── probes/                        # probe outputs (JSONL/CSV/log)
├── tests/                         # pytest suite
└── logs/                          # per-run game logs (JSONL)
```
