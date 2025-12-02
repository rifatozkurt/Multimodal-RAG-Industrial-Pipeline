""" Here I will set up multiple open source and/or proprietary LLMs for RAG
    
    I will use prompt templates to generate responses based on retrieved context.
    I will implement a simple RAG class and advanced RAG class, which take different llms as input.
    Each llm will have its own input prompt template.
    I will try to modularize the code as much as possible (i.e. same RAG class can take different llms as input).
    Models I plan to use:
    - Groq Qwen 3.2 32B
    - LlaVa 2 13B (or 7B)
    
    I will try to focus more on LLMs hosted on Groq and HF, also LLMs which use openai API, also LLMs using local inference (if time permits).
    
    In this cell I will set up LLMs and their prompt templates, and in the next cell I will set up RAG classes.
    
"""
import os
import torch
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import accelerate
from transformers.utils.import_utils import is_accelerate_available


groq_api_key = os.getenv("GROQ_API_KEY")


query = "testquery"
context = "testcontext"
def message_general(query, context):
    return [
        SystemMessage(content="You are a helpful assistant that provides concise answers."),
        HumanMessage(content=f"Use the following context to answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
    ]
    
def message_expander(query, length):
    return [
        SystemMessage(content="You are a helpful assistant that expands user queries into detailed, specific queries for document retrieval."),
        HumanMessage(content=f"Expand the following query to approximately {length} characters, adding relevant details and context to improve retrieval accuracy.\n\nQuery: {query}\n\nExpanded Query:"),
    ]
    
def message_cot(query):
    return [
        SystemMessage(content="You are a helpful assistant that expands user questions into step-by-step reasoning paths, asking intermediate questions to improve document retrieval accuracy."),
        HumanMessage(content=f"Expand the following question into a step-by-step reasoning path, formulating intermediate questions to guide document retrieval.\n\nQuestion: {query}\n\nReasoning Path:"),
    ]
    
def message_summarizer(message):
    return [
        SystemMessage(content="You are a helpful assistant that summarizes long contexts into concise summaries."),
        HumanMessage(content=f"Summarize the following context concisely, focusing on key points and relevant information.\n\nContext:\n{message}\n\nSummary:"),
    ]
    
def message_image_caption_generator(message: str):
    return [
        SystemMessage(
            content=(
                "You generate short, retrieval-focused captions for technical images "
                "from industrial manuals. The captions will be used as text queries "
                "for an image embedding model (like CLIP).\n"
                "Rules:\n"
                "- Describe what the image visually shows (e.g. time-series plot, wiring diagram, "
                "  block schematic, UI screenshot, parameter table).\n"
                "- Reuse important technical terms from the question (devices, components, signals) "
                "exactly as they appear.\n"
                "- Do NOT invent any new brand names, model numbers, part numbers, or specific instruments.\n"
                "- Do NOT make up numeric values, labels, or parameters not implied by the question.\n"
                "- Be specific but concise: 1 short sentence, max 2."
            )
        ),
        HumanMessage(
            content=(
                "Given the question below, write a single caption for an image from an industrial manual "
                "that would help answer it.\n\n"
                "The caption should:\n"
                "- Start by indicating the image type (e.g. 'time-series plot', 'wiring diagram', "
                "  'block diagram', 'UI screenshot', 'parameter table').\n"
                "- Include the key technical terms from the question so the image is clearly about the same system.\n"
                "- Avoid inventing brands, model numbers, or measurement devices that are not mentioned.\n"
                "- Avoid invented numeric values or detailed labels.\n\n"
                f"Question:\n{message}\n\n"
                "Image caption:"
            )
        ),
    ]



def get_groq_llm(api_key, model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=1024):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm

# helper functions to build inputs for different LLMs

def build_llava_inputs(
    processor_llava,
    query,
    text_context,
    image_paths,
    max_images = 1,
    device = "cuda:0"):
    
    """
    Build inputs for LLaVA
    Uses up to `max_images` from image_paths. The number of retrieved images (top_k) must be equal to this one.
    """
    # textual part
    text_content = (
        "You are a helpful assistant for industrial manuals.\n\n"
        "Use the following text context and image(s) to answer the question.\n\n"
        f"Text Context:\n{text_context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    # Open images 
    selected_paths = image_paths[:max_images]
    images = [Image.open(p).convert("RGB") for p in selected_paths]

    # Conversation in LLaVA format
    content = [{"type": "text", "text": text_content}]
    # LLaVA expects dummy "image" entries in the chat template, but real images are passed separately to the processor.
    for p in selected_paths:
        content.append({"type": "image"})

    conversation = [
        {
            "role": "user",
            "content": content,
        }
    ]

    prompt = processor_llava.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor_llava(images=images, text=prompt, return_tensors="pt").to(device)

    return inputs


def build_qwen_inputs(processor_qwen, query, text_context, image_paths, max_images = 1):
    """
    Build inputs for Qwen3-VL
    Up to `max_images` from image_paths (retrieved image number top_k must be equal to this one).
    """
    selected_paths = image_paths[:max_images]
    images = [Image.open(p).convert("RGB") for p in selected_paths]

    text_content = (
        "You are a helpful assistant for industrial manuals.\n\n"
        "Use the following text context and image(s) to answer the question.\n\n"
        f"Text Context:\n{text_context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    # Qwen3-VL expects messages with image objects directly
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": text_content})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = processor_qwen.apply_chat_template(messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt")

    return inputs



def load_llava_model(device="cuda:0"):
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(device)
    return model, processor



def load_qwen_model(dtype="auto", device="cuda:0"):
    model_qwen = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype=dtype, device_map=device)
    processor_qwen = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    return model_qwen, processor_qwen
    