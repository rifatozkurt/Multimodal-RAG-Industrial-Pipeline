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
from core.config import embedding_model_name, image_embedding_model_name, documents_path, vectordb_path, eval_dataset_path, eval_runs_path
from core.eval_utils import *

#-----------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

selected_model = "Qwen/Qwen3-VL-8B-Instruct"  # options: "Qwen/Qwen3-VL-8B-Instruct", "llava-hf/llava-v1.6-mistral-7b-hf"

preprocess_type = None   # or None / "chain_of_thought"
summarize = False
image_query_captioning = True

CHOICES = ["A", "B", "C", "D"]

#-----------------------------------------------------------------------

def main():
    print("="*60)
    print("MULTIMODAL RAG EVALUATION")
    print("="*60)
    print(f"Model: {selected_model}")
    print(f"Device: {device}")
    print(f"Preprocess type: {preprocess_type}")
    print(f"Summarize: {summarize}")
    print(f"Image query captioning: {image_query_captioning}")
    print("="*60)
    
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
    failed_questions = []

    for ex in tqdm(eval_data, desc=f"Evaluating {selected_model}"):
        try:
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
            
        except Exception as e:
            print(f"Error processing question {ex.get('id', 'unknown')}: {str(e)}")
            failed_questions.append({
                "id": ex.get("id", "unknown"),
                "error": str(e),
                "question": ex.get("question", "")[:100]
            })
            continue

    df = pd.DataFrame(results)
    
    # Report any failed questions
    if failed_questions:
        print(f"\nWARNING: {len(failed_questions)} questions failed to process:")
        for failed in failed_questions[:5]:  # Show first 5
            print(f"  - {failed['id']}: {failed['error']}")
        if len(failed_questions) > 5:
            print(f"  ... and {len(failed_questions) - 5} more")
    
    # metrics
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    total_questions = len(df)
    answered_questions = df["pred_answer"].notna().sum()
    unanswered_questions = total_questions - answered_questions
    
    print(f"Total questions: {total_questions}")
    print(f"Questions answered: {answered_questions} ({answered_questions/total_questions*100:.1f}%)")
    print(f"Questions unanswered: {unanswered_questions} ({unanswered_questions/total_questions*100:.1f}%)")
    
    overall_accuracy = df["is_correct"].mean()
    print(f"\nOverall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    
    # Accuracy among answered questions only
    if answered_questions > 0:
        mask = df["pred_answer"].notna()
        answered_accuracy = df.loc[mask, "is_correct"].mean()
        print(f"Accuracy (answered only): {answered_accuracy:.3f} ({answered_accuracy*100:.1f}%)")
        
        y_true = df.loc[mask, "gt_answer"]
        y_pred = df.loc[mask, "pred_answer"]

        print(f"\nClassification Report (answered questions only):")
        print(classification_report(y_true, y_pred, labels=CHOICES))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred, labels=CHOICES)
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CHOICES],
                                columns=[f"Pred_{c}" for c in CHOICES])
        print(cm_df)
    
    # Analysis by PDF
    print(f"\n" + "="*40)
    print("ANALYSIS BY PDF")
    print("="*40)
    pdf_analysis = df.groupby('pdf_name').agg({
        'is_correct': ['count', 'sum', 'mean'],
        'n_text_docs': 'mean',
        'n_image_docs': 'mean',
        'n_images_used': 'mean'
    }).round(3)
    pdf_analysis.columns = ['Total_Questions', 'Correct_Answers', 'Accuracy', 
                           'Avg_Text_Docs', 'Avg_Image_Docs', 'Avg_Images_Used']
    print(pdf_analysis)
    
    # Analysis by retrieval statistics
    print(f"\n" + "="*40)
    print("RETRIEVAL STATISTICS")
    print("="*40)
    print(f"Average text documents retrieved: {df['n_text_docs'].mean():.2f}")
    print(f"Average image documents retrieved: {df['n_image_docs'].mean():.2f}")
    print(f"Average images used in generation: {df['n_images_used'].mean():.2f}")
    
    # Performance by number of images used
    print(f"\nAccuracy by number of images used:")
    image_analysis = df.groupby('n_images_used')['is_correct'].agg(['count', 'mean']).round(3)
    image_analysis.columns = ['Question_Count', 'Accuracy']
    print(image_analysis)
    
    # Show some examples
    print(f"\n" + "="*40)
    print("SAMPLE RESULTS")
    print("="*40)
    print("Correct predictions:")
    correct_samples = df[df['is_correct'] == 1].head(3)
    for _, row in correct_samples.iterrows():
        print(f"Q: {row['question'][:100]}...")
        print(f"GT: {row['gt_answer']} | Pred: {row['pred_answer']} | Raw: {row['raw_output'][:50]}...")
        print()
    
    print("Incorrect predictions:")
    incorrect_samples = df[df['is_correct'] == 0].head(3)
    for _, row in incorrect_samples.iterrows():
        print(f"Q: {row['question'][:100]}...")
        print(f"GT: {row['gt_answer']} | Pred: {row['pred_answer']} | Raw: {row['raw_output'][:50]}...")
        print()

    # Create output directory
    output_dir = Path(eval_runs_path)
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    output_csv = output_dir / f"{selected_model.replace('/', '_')}_multimodal_rag_results.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Detailed results saved to: {output_csv}")
    
    # Save summary metrics
    summary_metrics = {
        'model': selected_model,
        'preprocess_type': preprocess_type,
        'total_questions': int(total_questions),
        'answered_questions': int(answered_questions),
        'overall_accuracy': float(overall_accuracy),
        'answered_accuracy': float(answered_accuracy) if answered_questions > 0 else None,
        'avg_text_docs': float(df['n_text_docs'].mean()),
        'avg_image_docs': float(df['n_image_docs'].mean()),
        'avg_images_used': float(df['n_images_used'].mean())
    }
    
    summary_file = output_dir / f"{selected_model.replace('/', '_')}_summary_metrics.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Summary metrics saved to: {summary_file}")
    
    # Save failed questions if any
    if failed_questions:
        failed_file = output_dir / f"{selected_model.replace('/', '_')}_failed_questions.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_questions, f, indent=2)
        print(f"Failed questions saved to: {failed_file}")
    
    print("="*60)
    print("EVALUATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()