import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
from pathlib import Path
from tqdm import tqdm

CHOICES = ["A", "B", "C", "D"]

def answer_question_with_qwen(example, mm_rag, verbose=False, backend="qwen_vl", preprocess_type=None):
    """
    Run the AdvancedMultimodalRAG pipeline for a single example.
    Returns:
        raw_answer_text (str)
        full_result_dict (dict from AdvancedMultimodalRAG.generate_response)
    """
    question_text = example["question"]

    if verbose:
        print("Question ID:", example.get("id"))
        print(question_text)

    result = mm_rag.generate_response(
        query=question_text,
        backend=backend,
        top_k_text=3,
        top_k_image=3,
        match_threshold_text=-0.5,
        match_threshold_image=-0.6,
        max_images=3,
        max_new_tokens=64,
        preprocess_type=preprocess_type,   # configurable preprocessing
        summarize=False,
    )

    raw_answer = result["answer"]
    return raw_answer, result


# Regex-based parser to extract a letter choice A–D from Qwen's answer
def extract_choice_letter(answer_text):
    """
    Try to extract a multiple-choice letter (A/B/C/D) from the model's output.
    Returns:
        "A"/"B"/"C"/"D" or None if nothing is found.
    """
    if not answer_text:
        return None

    txt = answer_text.strip()

    # 1) Look for patterns like "Answer: B" or "Answer - C"
    m = re.search(r"Answer\s*[:\-]\s*([ABCD])", txt, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2) Look for things like "B)" or "B." at the start of a line
    m = re.search(r"^\s*([ABCD])[\)\.]\s", txt, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).upper()

    # 3) Fallback: first standalone A-D token anywhere
    m = re.search(r"\b([ABCD])\b", txt, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Nothing found
    return None


def evaluate_on_dataset(dataset_path, mm_rag, verbose=False, output_path=None):
    """
    Evaluate the multimodal RAG pipeline on the dataset at dataset_path.
    Returns:
        accuracy (float)
        detailed_results_df (pd.DataFrame)
    """
    
    eval_path = Path(dataset_path)
    assert eval_path.exists(), f"Dataset not found at {eval_path}"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if verbose:
        print("Eval dataset path:", dataset_path)
        print(f"Loaded {len(dataset)} evaluation examples.")
        print("Example item:\n", dataset[0])

    results = []
    for example in tqdm(dataset, desc="Evaluating examples"):
        raw_answer, full_result = answer_question_with_qwen(example, mm_rag, verbose=verbose)
        predicted_choice = extract_choice_letter(raw_answer)
        correct_choice = example.get("answer")

        is_correct = (predicted_choice == correct_choice)

        results.append({
            "id": example["id"],
            "pdf_name": example["pdf_name"],
            "pdf_path": example["pdf_path"],
            "page": example["page"],
            "question": example["question"],
            "gt_answer": example["answer"],
            "pred_answer": predicted_choice,
            "is_correct": is_correct,
            "raw_output": raw_answer,

            # retrieval info
            "n_text_docs": len(full_result.get("retrieved_text_docs", [])),
            "n_image_docs": len(full_result.get("retrieved_image_docs", [])),
            "n_images_used": len(full_result.get("image_paths_used", [])),
        })

    results_df = pd.DataFrame(results)
    accuracy = accuracy_score(results_df["correct_choice"], results_df["predicted_choice"])

    if verbose:
        print("Classification Report:")
        print(classification_report(results_df["correct_choice"], results_df["predicted_choice"], labels=CHOICES))
        print("Confusion Matrix:")
        print(confusion_matrix(results_df["correct_choice"], results_df["predicted_choice"], labels=CHOICES))
        
    if output_path:
        results_df.to_csv(output_path, index=False,encoding="utf-8")
        if verbose:
            print(f"Saved detailed results to {output_path}")

    return accuracy, results_df
