import os
import torch
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from core.models_llm import groq_api_key, message_general, message_expander, message_cot, message_summarizer, message_image_caption_generator, load_llava_model, get_groq_llm, load_qwen_model, build_llava_inputs, build_qwen_inputs
from core.rag_pipelines import AdvancedMultimodalRAG
from core.retrievers import RetrieverMultiModal, RetrieverMultiModal_experimental
from core.vectordb import VectorDBManager
from core.embedders import EmbeddingManager, EmbeddingManager_Image
from core.config import embedding_model_name, image_embedding_model_name, documents_path, vectordb_path, eval_dataset_path
from core.eval_utils import *

#-----------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

selected_model = "Qwen/Qwen3-VL-8B-Instruct"  # options: "Qwen/Qwen3-VL-8B-Instruct", "llava-hf/llava-v1.6-mistral-7b-hf"

preprocess_type = "expand"   # or None / "chain_of_thought"
summarize = False
image_query_captioning = False

CHOICES = ["A", "B", "C", "D"]

#-----------------------------------------------------------------------

def main():
    print("Eval dataset path:", eval_dataset_path)

    eval_path = Path(eval_dataset_path)
    assert eval_path.exists(), f"Dataset not found at {eval_path}"

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"Loaded {len(eval_data)} evaluation examples.")
    print("Example item:\n", eval_data[0])
    
    
    if selected_model == "Qwen/Qwen3-VL-8B-Instruct":
        model_qwen, processor_qwen = load_qwen_model(device=device)
        model_llava, processor_llava = None, None
    elif selected_model == "llava-hf/llava-v1.6-mistral-7b-hf":
        model_llava, processor_llava = load_llava_model(device=device)
        model_qwen, processor_qwen = None, None

    if preprocess_type is not None:
        llm_llama31 = get_groq_llm(api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    else:
        llm_llama31 = None


    embedding_manager_txt = EmbeddingManager(model_name=embedding_model_name)
    embedding_manager_images = EmbeddingManager_Image(model_name=image_embedding_model_name)

    vector_db_manager_pdf = VectorDBManager(collection_name="pdf_documents_db",
                                        directory=os.path.join(vectordb_path, "pdf_db/"),
                                        source_type="pdf")
    vector_db_manager_pdf_images = VectorDBManager(collection_name="pdf_image_documents_db",
                                        directory=os.path.join(vectordb_path, "pdf_image_db/"),
                                        source_type="pdf_image")

    retriever_multimodal_image = RetrieverMultiModal_experimental(vector_db_text=vector_db_manager_pdf,
                                vector_db_image=vector_db_manager_pdf_images,
                                embedding_manager_text=embedding_manager_txt,
                                embedding_manager_image=embedding_manager_images)

    mm_rag = AdvancedMultimodalRAG(
        retriever_multimodal=retriever_multimodal_image,
        model_qwen=model_qwen,
        model_llava=model_llava,
        processor_llava=processor_llava,
        processor_qwen=processor_qwen,
        text_llm=llm_llama31,
    )

    backend = "qwen_vl" if selected_model == "Qwen/Qwen3-VL-8B-Instruct" else "llava"

    results = []

    for ex in tqdm(eval_data, desc="Evaluating Qwen-VL RAG"):
        gt = ex["answer"].strip().upper()

        raw_answer, full = answer_question_with_qwen(ex, mm_rag, verbose=False, backend=backend, preprocess_type=preprocess_type)
        pred = extract_choice_letter(raw_answer)

        is_correct = int(pred == gt) if pred else 0

        results.append({
            "id": ex["id"],
            "pdf_name": ex["pdf_name"],
            "pdf_path": ex["pdf_path"],
            "page": ex["page"],
            "question": ex["question"],
            "gt_answer": gt,
            "pred_answer": pred,
            "is_correct": is_correct,
            "raw_output": raw_answer,

            # retrieval info
            "n_text_docs": len(full.get("retrieved_text_docs", [])),
            "n_image_docs": len(full.get("retrieved_image_docs", [])),
            "n_images_used": len(full.get("image_paths_used", [])),
        })

    df = pd.DataFrame(results)
    df.head()
    # metrics
    print("Accuracy:", df["is_correct"].mean())

    mask = df["pred_answer"].notna()
    y_true = df.loc[mask, "gt_answer"]
    y_pred = df.loc[mask, "pred_answer"]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=CHOICES))

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=CHOICES)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CHOICES],
                            columns=[f"pred_{c}" for c in CHOICES])
    cm_df

    output_csv = "qwen_vl_multimodal_rag_results.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print("Saved:", output_csv)


if __name__ == "__main__":
    main()