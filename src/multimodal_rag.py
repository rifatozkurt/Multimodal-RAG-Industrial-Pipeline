import os
import torch
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from core.models_llm import groq_api_key, message_general, message_expander, message_cot, message_summarizer, message_image_caption_generator, load_llava_model, get_groq_llm, load_qwen_model, build_llava_inputs, build_qwen_inputs
from core.rag_pipelines import AdvancedMultimodalRAG, BitsAndBytesConfig
from core.retrievers import RetrieverMultiModal, RetrieverMultiModal_experimental
from core.vectordb import VectorDBManager
from core.embedders import EmbeddingManager, EmbeddingManager_Image
from core.config import embedding_model_name, image_embedding_model_name, documents_path, vectordb_path

#-----------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

selected_model = "Qwen/Qwen3-VL-8B-Instruct"  # options: "Qwen/Qwen3-VL-8B-Instruct", "llava-hf/llava-v1.6-mistral-7b-hf"

query = "What maintenance steps are recommended for the equipment shown in the images?"
image_paths = ["../documents/pdf/extracted_images/wildpaper_p1_x61_w858_h450.png",]

preprocess_type = "expand"   # or None / "chain_of_thought"
summarize = False
image_query_captioning = False

#-----------------------------------------------------------------------

def main():
    if selected_model == "Qwen/Qwen3-VL-8B-Instruct":
        model_qwen, processor_qwen = load_qwen_model(device=device)
        model_llava, processor_llava = None, None
    elif selected_model == "llava-hf/llava-v1.6-mistral-7b-hf":
        model_llava, processor_llava = load_llava_model(device=device)
        model_qwen, processor_qwen = None, None

    llm_llama31 = get_groq_llm(api_key=groq_api_key, model_name="groq/llama3-1.1b-chat", device=device)


    embedding_manager_txt = EmbeddingManager(model_name=embedding_model_name)
    embedding_manager_images = EmbeddingManager_Image(model_name=image_embedding_model_name)

    vector_db_manager_pdf = VectorDBManager.load_from_disk(os.path.join(vectordb_path, "pdf_db/"))
    vector_db_manager_pdf_images = VectorDBManager.load_from_disk(os.path.join(vectordb_path, "pdf_image_db/"))

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
        preprocess_type=preprocess_type,   # or None / "chain_of_thought"
        summarize=summarize,
        image_query_captioning=image_query_captioning,
    )
    print("Final Response:")
    print(result['answer'])
    print("Sources:")
    for src in result['sources']:
        print(src)
        
if __name__ == "__main__":
    main()