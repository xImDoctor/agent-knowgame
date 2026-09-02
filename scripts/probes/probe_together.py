"""Minimal single-call probe against Together API

Usage:
    python scripts/probes/probe_together.py [model_id]

Default model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo.
Verifies that TOGETHER_API_KEY works, the model is serverless,
and pydantic schema is accepted by Together's grammar engine
(there are problems with it using some models).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

from game.clients import TogetherLLMClient
from game.prompt_builder import AnswerResponse


DEFAULT_MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'


def main():
    load_dotenv()

    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f'Probing model: {model}')

    client = TogetherLLMClient(model=model)
    response = client.structured_call(
        [{'role': 'user', 'content': 'Output exactly: {"answer": "hello"}'}],
        AnswerResponse,
    )
    print(f'OK, parsed: {response}')


if __name__ == '__main__':
    main()
