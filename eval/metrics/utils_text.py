import re
from pathlib import Path


def strip_answer_prefix(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"^\s*answer\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()


def normalize_for_exact(text: str) -> str:
    text = strip_answer_prefix(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_tokens(text: str) -> str:
    text = strip_answer_prefix(text)
    text = text.lower()
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_for_tokens(text)
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    gold_counts = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1
    overlap = 0
    for tok, cnt in pred_counts.items():
        overlap += min(cnt, gold_counts.get(tok, 0))
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_choice(answer_text: str, choices: list[str] | None = None) -> str | None:
    if not answer_text:
        return None
    text = answer_text.strip()
    if choices is None:
        choices = ["A", "B", "C", "D"]
    normalized_choices = [c.strip().upper() for c in choices]

    m = re.search(r"answer\s*[:\-]\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if m:
        choice = m.group(1).upper()
        return choice if choice in normalized_choices else None

    m = re.search(r"^\s*([A-Z])[\)\.\:]\s*", text, flags=re.IGNORECASE)
    if m:
        choice = m.group(1).upper()
        return choice if choice in normalized_choices else None

    m = re.search(r"\b([A-Z])\b", text, flags=re.IGNORECASE)
    if m:
        choice = m.group(1).upper()
        return choice if choice in normalized_choices else None

    return None


def parse_number_list(text: str) -> list[float]:
    if not text:
        return []
    text = strip_answer_prefix(text)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            continue
    return numbers


def normalize_path(path_str: str) -> str:
    if not path_str:
        return ""
    path_str = path_str.replace("\\", "/").strip().lower()
    return path_str


def image_id_candidates(path_str: str) -> set[str]:
    if not path_str:
        return set()
    normalized = normalize_path(path_str)
    basename = Path(normalized).name
    return {normalized, basename}
