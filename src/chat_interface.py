"""
THIS IS CREATED BY CHATGPT, NOT TESTED YET.


Gradio UI for Advanced Multimodal RAG

Save as: multimodal_rag_ui.py
Run from repo root (or from src, depending on your imports):
    python multimodal_rag_ui.py
"""

import os
import torch
import gradio as gr

from core.models_llm import (
    groq_api_key,
    get_groq_llm,
    load_llava_model,
    load_qwen_model,
)
from core.rag_pipelines import AdvancedMultimodalRAG
from core.retrievers import RetrieverMultiModal_experimental
from core.vectordb import VectorDBManager
from core.embedders import EmbeddingManager, EmbeddingManager_Image
from core.config import (
    embedding_model_name,
    image_embedding_model_name,
    vectordb_path,
)

# ---------------------------------------------------------------------
# Helpers to build / cache the pipeline
# ---------------------------------------------------------------------

def build_pipeline(selected_model: str):
    """
    Build an AdvancedMultimodalRAG instance for the given backend.

    This loads:
      - text LLM (Groq)
      - Qwen3-VL OR LLaVA
      - text & image embedders
      - text & image vector DBs
      - multimodal retriever
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Vision-language model selection
    if selected_model == "Qwen/Qwen3-VL-8B-Instruct":
        model_qwen, processor_qwen = load_qwen_model(device=device)
        model_llava, processor_llava = None, None
        backend = "qwen_vl"
    else:  # "llava-hf/llava-v1.6-mistral-7b-hf"
        model_llava, processor_llava = load_llava_model(device=device)
        model_qwen, processor_qwen = None, None
        backend = "llava"

    # Text LLM from Groq (adjust model_name if needed)
    llm_llama31 = get_groq_llm(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    # Embedding managers
    embedding_manager_txt = EmbeddingManager(model_name=embedding_model_name)
    embedding_manager_images = EmbeddingManager_Image(
        model_name=image_embedding_model_name
    )

    # Load vector DBs from disk
    pdf_db_path = os.path.join(vectordb_path, "pdf_db/")
    pdf_image_db_path = os.path.join(vectordb_path, "pdf_image_db/")

    vector_db_manager_pdf = VectorDBManager(collection_name="pdf_documents_db",
                                        directory=os.path.join(vectordb_path, "pdf_db/"),
                                        source_type="pdf")
    vector_db_manager_pdf_images = VectorDBManager(collection_name="pdf_image_documents_db",
                                        directory=os.path.join(vectordb_path, "pdf_image_db/"),
                                        source_type="pdf_image")

    # Multimodal retriever
    retriever_multimodal_image = RetrieverMultiModal_experimental(
        vector_db_text=vector_db_manager_pdf,
        vector_db_image=vector_db_manager_pdf_images,
        embedding_manager_text=embedding_manager_txt,
        embedding_manager_image=embedding_manager_images,
    )

    # Advanced Multimodal RAG pipeline
    mm_rag = AdvancedMultimodalRAG(
        retriever_multimodal=retriever_multimodal_image,
        model_qwen=model_qwen,
        model_llava=model_llava,
        processor_llava=processor_llava,
        processor_qwen=processor_qwen,
        text_llm=llm_llama31,
    )

    return mm_rag, backend


# ---------------------------------------------------------------------
# Gradio chat function
# ---------------------------------------------------------------------

def chat_with_rag(
    message,
    history,
    selected_model,
    preprocess_type,
    summarize,
    image_query_captioning,
    top_k_text,
    top_k_image,
    max_images,
    max_tokens,
    state,
):
    """
    Gradio callback.

    - Builds (or reuses) the pipeline for the selected model.
    - Calls AdvancedMultimodalRAG.generate_response(...)
    - Updates chat history.
    """
    # Initialize state dict on first run
    if state is None:
        state = {}

    # (Re)build pipeline if not present or if model changed
    if (
        "mm_rag" not in state
        or "backend" not in state
        or state.get("selected_model") != selected_model
    ):
        mm_rag, backend = build_pipeline(selected_model)
        state["mm_rag"] = mm_rag
        state["backend"] = backend
        state["selected_model"] = selected_model
    else:
        mm_rag = state["mm_rag"]
        backend = state["backend"]

    # Convert preprocess_type "none" -> None
    if preprocess_type == "none":
        preprocess = None
    else:
        preprocess = preprocess_type

    # Call the pipeline
    try:
        result = mm_rag.generate_response(
            query=message,
            backend=backend,
            top_k_text=top_k_text,
            top_k_image=top_k_image,
            max_images=max_images,
            preprocess_type=preprocess,         # e.g. "expand", "chain_of_thought", or None
            summarize=summarize,                # bool
            image_query_captioning=image_query_captioning,  # bool
            max_new_tokens=max_tokens,             # assuming your method supports this
        )
        answer = result.get("answer", "[No answer returned]")
        print(f"Debug: result type = {type(result)}, answer = {answer}")
    except TypeError:
        # In case your generate_response() doesn't support max_tokens yet
        result = mm_rag.generate_response(
            query=message,
            backend=backend,
            top_k_text=top_k_text,
            top_k_image=top_k_image,
            max_images=max_images,
            preprocess_type=preprocess,
            summarize=summarize,
            image_query_captioning=image_query_captioning,
        )
        answer = result.get("answer", "[No answer returned]")
        print(f"Debug: result type = {type(result)}, answer = {answer}")

    # Update chat history - ensure answer is a string, not dict
    if isinstance(answer, dict):
        answer_text = answer.get("answer", "[No answer returned]")
    else:
        answer_text = str(answer) if answer is not None else "[No answer]"
    
    # Ensure history is in correct format for Gradio chatbot
    if history is None:
        history = []
    
    # Ensure both message and answer are strings and not empty
    message_str = str(message).strip() if message is not None else "Empty message"
    answer_str = str(answer_text).strip() if answer_text is not None else "No answer"
    
    # Ensure no empty strings which might cause Gradio issues
    if not message_str:
        message_str = "Empty message"
    if not answer_str:
        answer_str = "No answer provided"
    
    print(f"Debug: Adding to history - message: {message_str[:50]}..., answer: {answer_str[:50]}...")
    
    # Use messages format for newer Gradio versions (which seems to be required)
    history.append({"role": "user", "content": message_str})
    history.append({"role": "assistant", "content": answer_str})
    
    print(f"Debug: Final history length: {len(history)}")

    return history, state


# ---------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------

def create_interface():
    with gr.Blocks(title="Multimodal RAG Chat") as demo:
        gr.Markdown("# 🧠📚 Multimodal RAG Chat\nChat with your documents & images.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Multimodal RAG",
                    height=500,
                )
                user_input = gr.Textbox(
                    label="Your question",
                    placeholder="Ask something about your documents...",
                    value="how do drones localize themselves in the lack of gnss signals?",
                    lines=2,
                )
                send_btn = gr.Button("Send")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                selected_model = gr.Dropdown(
                    choices=[
                        "Qwen/Qwen3-VL-8B-Instruct",
                        "llava-hf/llava-v1.6-mistral-7b-hf",
                    ],
                    value="Qwen/Qwen3-VL-8B-Instruct",
                    label="Backend model",
                )

                preprocess_type = gr.Radio(
                    choices=["none", "expand", "chain_of_thought"],
                    value="none",
                    label="Query preprocessing",
                )

                summarize = gr.Checkbox(
                    value=False,
                    label="Summarize long answers",
                )

                image_query_captioning = gr.Checkbox(
                    value=False,
                    label="Enable image query captioning",
                )

                top_k_text = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="top_k_text",
                )

                top_k_image = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=2,
                    step=1,
                    label="top_k_image",
                )

                max_images = gr.Slider(
                    minimum=0,
                    maximum=5,
                    value=2,
                    step=1,
                    label="max_images",
                )

                max_tokens = gr.Slider(
                    minimum=32,
                    maximum=2048,
                    value=512,
                    step=32,
                    label="Max tokens (LLM output)",
                )

        # Gradio state to cache pipeline & backend
        state = gr.State(value=None)

        # Wire the button and Enter key
        def send_message_and_clear():
            return ""
        
        send_event = send_btn.click(
            fn=chat_with_rag,
            inputs=[
                user_input,
                chatbot,
                selected_model,
                preprocess_type,
                summarize,
                image_query_captioning,
                top_k_text,
                top_k_image,
                max_images,
                max_tokens,
                state,
            ],
            outputs=[chatbot, state],
        )
        
        # Also enable Enter key submission
        user_input.submit(
            fn=chat_with_rag,
            inputs=[
                user_input,
                chatbot,
                selected_model,
                preprocess_type,
                summarize,
                image_query_captioning,
                top_k_text,
                top_k_image,
                max_images,
                max_tokens,
                state,
            ],
            outputs=[chatbot, state],
        )

        # Clear user input after sending (both button and Enter)
        send_btn.click(
            fn=send_message_and_clear,
            inputs=None,
            outputs=user_input,
        )
        
        user_input.submit(
            fn=send_message_and_clear,
            inputs=None,
            outputs=user_input,
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    demo.launch()
