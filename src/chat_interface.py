"""
Gradio UI for Advanced Multimodal RAG

Features:
- Multimodal RAG with Qwen3-VL and LLaVA models
- Model preloading for faster response times
- Query preprocessing and image captioning
- Adjustable parameters for retrieval and generation
"""

import os
import torch
import gradio as gr
import time
import tempfile
import shutil
from pathlib import Path

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


share_option = True  


def build_pipeline(selected_model: str, progress=None):
    """
    Build an AdvancedMultimodalRAG instance for the given backend.
    
    Args:
        selected_model: Model name to load
        progress: Optional Gradio progress bar
    
    Returns:
        tuple: (mm_rag, backend)
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if progress:
        progress(0.1, desc="Loading vision-language model...")
    
    # Vision-language model selection
    if selected_model == "Qwen/Qwen3-VL-8B-Instruct":
        model_qwen, processor_qwen = load_qwen_model(device=device)
        model_llava, processor_llava = None, None
        backend = "qwen_vl"
    else:  # "llava-hf/llava-v1.6-mistral-7b-hf"
        model_llava, processor_llava = load_llava_model(device=device)
        model_qwen, processor_qwen = None, None
        backend = "llava"

    if progress:
        progress(0.4, desc="Loading text LLM from Groq...")
    
    # Text LLM from Groq
    llm_llama31 = get_groq_llm(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    if progress:
        progress(0.6, desc="Loading embedding models...")
    
    # Embedding managers
    embedding_manager_txt = EmbeddingManager(model_name=embedding_model_name)
    embedding_manager_images = EmbeddingManager_Image(
        model_name=image_embedding_model_name
    )

    if progress:
        progress(0.8, desc="Loading vector databases...")
    
    # Load vector DBs from disk
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

    if progress:
        progress(0.9, desc="Initializing RAG pipeline...")
    
    # Advanced Multimodal RAG pipeline
    mm_rag = AdvancedMultimodalRAG(
        retriever_multimodal=retriever_multimodal_image,
        model_qwen=model_qwen,
        model_llava=model_llava,
        processor_llava=processor_llava,
        processor_qwen=processor_qwen,
        text_llm=llm_llama31,
    )
    
    if progress:
        progress(1.0, desc="Model loaded successfully!")

    return mm_rag, backend


def preload_model(selected_model, progress=gr.Progress()):
    """
    Preload the selected model and return status message.
    """
    try:
        start_time = time.time()
        mm_rag, backend = build_pipeline(selected_model, progress)
        end_time = time.time()
        
        load_time = end_time - start_time
        status_msg = f"{selected_model} loaded successfully in {load_time:.1f}s and ready to use!"
        
        # Return the loaded pipeline components and success status
        return {
            "mm_rag": mm_rag,
            "backend": backend,
            "selected_model": selected_model,
            "loaded": True,
            "load_time": load_time
        }, status_msg, gr.update(interactive=True)  # Enable send button
        
    except Exception as e:
        error_msg = f"Error loading {selected_model}: {str(e)}"
        return {
            "loaded": False,
            "error": str(e)
        }, error_msg, gr.update(interactive=False)  # Keep send button disabled



def save_uploaded_image(uploaded_file):
    """
    Save uploaded image to a temporary location and return the path.
    """
    if uploaded_file is None:
        return None
    
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "multimodal_rag_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        # Get original filename and extension
        original_name = Path(uploaded_file.name).name
        temp_path = temp_dir / f"uploaded_{int(time.time())}_{original_name}"
        
        # Copy uploaded file to temp location
        shutil.copy2(uploaded_file.name, temp_path)
        
        print(f"Saved uploaded image to: {temp_path}")
        return str(temp_path)
        
    except Exception as e:
        print(f"Error saving uploaded image: {e}")
        return None

def clear_uploaded_image():
    """Clear the image upload component."""
    return None



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
    uploaded_image,
):
    """
    Gradio callback for chat functionality.
    
    Uses preloaded model if available, otherwise shows error message.
    """
    # Check if message is empty
    if not message or not message.strip():
        return history, state
    
    # Initialize state dict on first run
    if state is None:
        state = {}
    
    # Check if model is preloaded
    if not state.get("loaded", False):
        # Add error message to history
        error_msg = "Please load a model first using the 'Load Model' button above."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, state
    
    # Check if the selected model matches the loaded model
    if state.get("selected_model") != selected_model:
        error_msg = f"Model mismatch! Currently loaded: {state.get('selected_model', 'None')}, Selected: {selected_model}. Please reload the model."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, state
    
    # Get preloaded components
    mm_rag = state["mm_rag"]
    backend = state["backend"]

    # Handle uploaded image
    uploaded_image_path = None
    if uploaded_image is not None:
        uploaded_image_path = save_uploaded_image(uploaded_image)
        if uploaded_image_path:
            print(f"Using uploaded image: {uploaded_image_path}")
            # Ensure we use at least 1 image if one is uploaded
            max_images = max(max_images, 1)

    # Convert preprocess_type "none" -> None
    if preprocess_type == "none":
        preprocess = None
    else:
        preprocess = preprocess_type

    # Enhanced approach: Single-generation with uploaded image prioritization
    try:
        if uploaded_image_path:
            # Use custom pipeline that integrates uploaded image before generation
            # This runs generation only once while prioritizing the uploaded image
            result = generate_response_with_uploaded_image(
                mm_rag=mm_rag,
                query=message,
                uploaded_image_path=uploaded_image_path,
                backend=backend,
                top_k_text=top_k_text,
                top_k_image=top_k_image,
                max_images=max_images,
                preprocess_type=preprocess,
                summarize=summarize,
                image_query_captioning=image_query_captioning,
                max_new_tokens=max_tokens,
            )
        else:
            # Standard RAG pipeline when no image is uploaded
            result = mm_rag.generate_response(
                query=message,
                backend=backend,
                top_k_text=top_k_text,
                top_k_image=top_k_image,
                max_images=max_images,
                preprocess_type=preprocess,
                summarize=summarize,
                image_query_captioning=image_query_captioning,
                max_new_tokens=max_tokens,
            )
        
        answer = result.get("answer", "[No answer returned]")
        print(f"Debug: Generated answer successfully")
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
        print(f"Debug: Generated answer successfully (fallback)")
    except Exception as e:
        answer = f"Error generating response: {str(e)}"
        print(f"Error in chat_with_rag: {e}")

    # Update chat history - ensure answer is a string
    if isinstance(answer, dict):
        answer_text = answer.get("answer", "[No answer returned]")
    else:
        answer_text = str(answer) if answer is not None else "[No answer]"
    
    # Ensure both message and answer are strings and not empty
    message_str = str(message).strip() if message is not None else "Empty message"
    answer_str = str(answer_text).strip() if answer_text is not None else "No answer"
    
    # Ensure no empty strings
    if not message_str:
        message_str = "Empty message"
    if not answer_str:
        answer_str = "No answer provided"
    
    # Add to chat history using messages format
    history.append({"role": "user", "content": message_str})
    history.append({"role": "assistant", "content": answer_str})
    
    print(f"Debug: Added exchange to history, total length: {len(history)}")

    return history, state


def generate_response_with_uploaded_image(
    mm_rag,
    query,
    uploaded_image_path,
    backend,
    top_k_text=3,
    top_k_image=2,
    max_images=2,
    preprocess_type=None,
    summarize=False,
    image_query_captioning=False,
    max_new_tokens=256,
):
    """
    Enhanced RAG generation that prioritizes uploaded images.
    
    This function modifies the standard RAG pipeline to:
    1. Run normal text/image retrieval
    2. Inject uploaded image as highest priority
    3. Generate response once with prioritized image list
    """
    print(f"Generating response with uploaded image: {uploaded_image_path}")
    
    # Step 1: Run retrieval (same as normal pipeline)
    query_retrieval = mm_rag._preprocess_query(
        query=query,
        preprocess_type=preprocess_type,
        chunk_size=1000
    )
    
    # Handle image query with CLIP token limit safety
    if image_query_captioning:
        query_image_retrieval = mm_rag._preprocess_image_query(query=query)
    elif len(query) > 200:
        print(f"Query too long for CLIP ({len(query)} chars). Auto-generating image caption...")
        query_image_retrieval = mm_rag._preprocess_image_query(query=query)
    else:
        query_image_retrieval = query
    
    # Step 2: Retrieve documents from vector databases
    try:
        text_docs, image_docs = mm_rag.retriever_multimodal.retrieve(
            query=query_retrieval,
            query_image=query_image_retrieval,
            top_k_text=top_k_text,
            top_k_image=top_k_image,
            match_threshold_text=-0,
            match_threshold_image=-0
        )
    except TypeError as e:
        # Handle case where retriever doesn't support query_image parameter
        if "query_image" in str(e):
            text_docs, image_docs = mm_rag.retriever_multimodal.retrieve(
                query=query_retrieval,
                top_k_text=top_k_text,
                top_k_image=top_k_image,
                match_threshold_text=-0,
                match_threshold_image=-0
            )
        else:
            raise
    
    # Step 3: Inject uploaded image as highest priority
    uploaded_image_doc = {
        "id": "uploaded_image",
        "score": 1.0,  # Highest priority score
        "metadata": {
            "image_path": uploaded_image_path,
            "source": "user_upload",
            "page_text": f"User uploaded image for analysis with query: {query}"
        },
        "document": f"User uploaded image: {uploaded_image_path}"
    }
    
    # Prioritize uploaded image and adjust retrieved images accordingly
    if max_images > 0:
        image_docs = [uploaded_image_doc] + image_docs[:max_images-1]
        print(f"Prioritized uploaded image. Using {len(image_docs)} total images.")
    
    # Step 4: Build text context (same as normal pipeline)
    text_context = mm_rag._build_text_context(
        text_docs=text_docs,
        image_docs=image_docs,
        max_text_chunks=top_k_text
    )
    
    # Step 5: Select image paths for generation
    image_paths_used = mm_rag._select_image_paths(
        image_docs=image_docs,
        max_images=max_images
    )
    
    # Step 6: Run generation once with prioritized images
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if backend == "llava":
        answer = mm_rag._run_llava(
            query=query,
            text_context=text_context,
            image_paths=image_paths_used,
            max_images=max_images,
            max_new_tokens=max_new_tokens,
            device=device
        )
    elif backend == "qwen_vl":
        answer = mm_rag._run_qwen(
            query=query,
            text_context=text_context,
            image_paths=image_paths_used,
            max_images=max_images,
            max_new_tokens=max_new_tokens
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Step 7: Optional summarization
    if summarize and mm_rag.text_llm is not None:
        print("Summarizing answer with text LLM...")
        from core.models_llm import message_summarizer
        summarization_prompt = message_summarizer(message=answer)
        summary_response = mm_rag.text_llm.invoke(summarization_prompt)
        answer = summary_response.content
        print("Summarized answer completed.")
    
    # Step 8: Build result (same format as normal pipeline)
    sources_text = []
    for doc in text_docs:
        sources_text.append({
            "id": doc["id"],
            "score": doc["score"],
            "metadata": doc["metadata"],
        })

    sources_image = []
    for doc in image_docs:
        p = doc["metadata"].get("image_path") or doc["metadata"].get("page_image_path")
        sources_image.append({
            "id": doc["id"],
            "score": doc["score"],
            "metadata": doc["metadata"],
            "image_path": p,
        })

    result = {
        "answer": answer,
        "backend": backend,
        "query_original": query,
        "query_retrieval": query_retrieval,
        "text_context": text_context,
        "image_paths_used": image_paths_used,
        "retrieved_text_docs": text_docs,
        "retrieved_image_docs": image_docs,
        "sources_text": sources_text,
        "sources_image": sources_image,
        "message_history": mm_rag.message_history,
    }

    mm_rag.message_history.append(result)
    return result


# ---------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------

def create_interface():
    with gr.Blocks(title="Multimodal RAG Chat") as demo:
        gr.Markdown("# Multimodal RAG Chat\nChat with your documents & images using AI vision-language models.")

        with gr.Row():
            with gr.Column(scale=3):
                # Initialize chatbot with welcome message
                initial_history = [
                    {
                        "role": "assistant", 
                        "content": """ Welcome to Multimodal RAG Chat!

**Getting Started:**
1. **Load a Model**: Click 'Load Model' button to preload your chosen vision-language model
2. **Ask Questions**: Once loaded, ask questions about your documents and images
3. **Upload Images**: Optionally upload an image to ask questions about it specifically
4. **Adjust Settings**: Tweak parameters in the sidebar for optimal results

**Tips:**
- Try questions about drone localization, UAV navigation, or technical documentation
- Upload images (PNG, JPG, etc.) to ask questions about specific diagrams or photos
- Enable 'Query Preprocessing' for better retrieval from document database
- Use 'Image Query Captioning' for better image search in your document collection
- Adjust top_k values to get more/fewer relevant documents

**Example Questions:**
- Upload a technical diagram and ask: "Explain what this diagram shows"
- "How do drones localize themselves without GNSS signals?"
- "What maintenance steps are recommended for the equipment?"
- Upload a photo and ask: "What components do you see in this image?"

Ready to explore your documents? Load a model to get started! """
                    }
                ]
                
                chatbot = gr.Chatbot(
                    label="Multimodal RAG Assistant",
                    height=500,
                    value=initial_history,
                    type="messages"
                )
                
                with gr.Row():
                    user_input = gr.Textbox(
                        label="Your question",
                        placeholder="Ask something about your documents... (Load a model first!)",
                        value="",
                        lines=2,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", interactive=False, scale=1)
                
                with gr.Row():
                    image_upload = gr.File(
                        label="Upload Image (Optional)",
                        file_types=["image"],
                        file_count="single"
                    )
                    clear_image_btn = gr.Button("Clear Image", variant="secondary", size="sm")

            with gr.Column(scale=1):
                gr.Markdown("### Model Configuration")
                
                selected_model = gr.Dropdown(
                    choices=[
                        "Qwen/Qwen3-VL-8B-Instruct",
                        "llava-hf/llava-v1.6-mistral-7b-hf",
                    ],
                    value="Qwen/Qwen3-VL-8B-Instruct",
                    label="Vision-Language Model",
                    info="Choose your multimodal AI model"
                )
                
                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="primary", size="lg")
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value="No model loaded. Click 'Load Model' to start.",
                    interactive=False,
                    lines=2,
                )
                
                gr.Markdown("### Generation Settings")

                preprocess_type = gr.Radio(
                    choices=["none", "expand", "chain_of_thought"],
                    value="none",
                    label="Query Preprocessing",
                    info="Enhance queries for better retrieval"
                )

                with gr.Row():
                    summarize = gr.Checkbox(
                        value=False,
                        label="Summarize Answers",
                        info="Make long answers more concise"
                    )
                    image_query_captioning = gr.Checkbox(
                        value=False,
                        label="Image Query Captioning",
                        info="Generate image captions for better image retrieval"
                    )

                gr.Markdown("### Retrieval Settings")

                with gr.Row():
                    top_k_text = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Text Documents",
                        info="Number of text chunks to retrieve"
                    )
                    top_k_image = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=1,
                        label="Image Documents", 
                        info="Number of images to retrieve"
                    )

                with gr.Row():
                    max_images = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=2,
                        step=1,
                        label="Max Images Used",
                        info="Maximum images for generation"
                    )
                    max_tokens = gr.Slider(
                        minimum=32,
                        maximum=2048,
                        value=256,
                        step=32,
                        label="Max Output Tokens",
                        info="Maximum length of generated response"
                    )

        # Gradio state to cache pipeline & backend  
        state = gr.State(value={"loaded": False})

        # Model loading functionality
        load_btn.click(
            fn=preload_model,
            inputs=[selected_model],
            outputs=[state, model_status, send_btn],
            show_progress=True
        )

        # Chat functionality
        def send_message_and_clear():
            return ""
        
        # Wire the send button
        send_btn.click(
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
                image_upload,
            ],
            outputs=[chatbot, state],
        )
        
        # Enable Enter key submission
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
                image_upload,
            ],
            outputs=[chatbot, state],
        )

        # Clear input after sending (both button and Enter)
        send_btn.click(fn=send_message_and_clear, outputs=user_input)
        user_input.submit(fn=send_message_and_clear, outputs=user_input)
        
        # Clear image functionality
        clear_image_btn.click(fn=clear_uploaded_image, outputs=image_upload)
        
        # Update placeholder text when model is loaded
        def update_placeholder(state):
            if state and state.get("loaded", False):
                return gr.update(placeholder="Ask something about your documents...")
            else:
                return gr.update(placeholder="Load a model first to start chatting!")
        
        # Update UI when state changes
        state.change(fn=update_placeholder, inputs=state, outputs=user_input)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    demo.launch(share=share_option)
