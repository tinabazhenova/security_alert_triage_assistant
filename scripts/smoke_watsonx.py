"""Manual smoke check: one Granite call and one embedding call against watsonx.

Needs real credentials in .env (see .env.example). Not run in CI and not
collected by pytest (testpaths = ["tests"]).

Usage:
    python scripts/smoke_watsonx.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage.config import ConfigError
from triage.llm import get_embeddings, get_llm


def main() -> int:
    load_dotenv(ROOT / ".env")

    try:
        llm = get_llm()
        embeddings = get_embeddings()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    print("LLM call...")
    answer = llm.invoke("In one sentence, what is a SIEM alert?")
    print(f"  -> {answer.strip()}")

    print("Embedding call...")
    vector = embeddings.embed_query("Suspicious PowerShell execution on host WS-042")
    print(f"  -> dimension {len(vector)}, first values {vector[:3]}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
