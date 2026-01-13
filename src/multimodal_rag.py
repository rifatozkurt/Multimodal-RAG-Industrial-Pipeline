import os
import torch
from core.models_llm import groq_api_key, load_llava_model, get_groq_llm, load_qwen_model
from core.rag_pipelines import AdvancedMultimodalRAG
from core.retrievers import RetrieverMultiModal, RetrieverMultiModal_experimental
from core.vectordb import VectorDBManager
from core.embedders import EmbeddingManager, EmbeddingManager_Image
from core.config import embedding_model_name, image_embedding_model_name, documents_path, vectordb_path

#-----------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
selected_model = "Qwen/Qwen3-VL-8B-Instruct"  # options: "Qwen/Qwen3-VL-8B-Instruct", "llava-hf/llava-v1.6-mistral-7b-hf"

query = "The following is a question about an industrial device. Give only the letter choice in your answer.\n\nQuestion: In an SR latch built from NOR gates, which condition is not allowed\nA.S=0, R=0\nB.S=0, R=1\nC.S=1, R=0\nD.S=1, R=1\nAnswer:"

preprocess_type = None   # or None / "chain_of_thought"
summarize = False
image_query_captioning = False
max_new_tokens = 256

#-----------------------------------------------------------------------

def main():
    # Initialize model variables
    model_qwen, processor_qwen = None, None
    model_llava, processor_llava = None, None
    
    if selected_model == "Qwen/Qwen3-VL-8B-Instruct":
        model_qwen, processor_qwen = load_qwen_model(device=device)
    elif selected_model == "llava-hf/llava-v1.6-mistral-7b-hf":
        model_llava, processor_llava = load_llava_model(device=device)
    else:
        raise ValueError(f"Unsupported model: {selected_model}")

    llm_llama31 = get_groq_llm(api_key=groq_api_key, model_name="llama-3.1-8b-instant")


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

    result = mm_rag.generate_response(
        query= query,
        backend=backend,
        top_k_text=3,
        top_k_image=2,
        max_images=2,
        match_threshold_text=-0.5,
        match_threshold_image=-1,
        preprocess_type=preprocess_type,   # or None / "chain_of_thought"
        summarize=summarize,
        image_query_captioning=image_query_captioning,
        max_new_tokens=max_new_tokens
    )
    print("Final Response:")
    print(result['answer'])
    print("Text Sources:")
    for src in result['sources_text']:
        print(src)
    print("Image Sources:")
    for src in result['sources_image']:
        #only print image source paths
        print(src['metadata']['image_path'])

if __name__ == "__main__":
    main()