import sys
from pathlib import Path

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from core.config import nli_model_name  # noqa: E402


def main() -> None:
    print("Caching ROUGE metric...")
    evaluate.load("rouge")

    print(f"Caching NLI model: {nli_model_name}")
    AutoTokenizer.from_pretrained(nli_model_name)
    AutoModelForSequenceClassification.from_pretrained(nli_model_name)

    print("Done.")


if __name__ == "__main__":
    main()
