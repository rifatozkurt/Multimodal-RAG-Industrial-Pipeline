import os
import torch
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from core.models_llm import groq_api_key, message_general, message_expander, message_cot, message_summarizer, message_image_caption_generator, load_llava_model, get_groq_llm, load_qwen_model, build_llava_inputs, build_qwen_inputs

device = "cuda:0" if torch.cuda.is_available() else "cpu"

selected_model = "Qwen/Qwen3-VL-8B-Instruct"  # options: "Qwen/Qwen3-VL-8B-Instruct", "llava-hf/llava-v1.6-mistral-7b-hf"

query = "What maintenance steps are recommended for the equipment shown in the images?"
image_paths = ["../documents/pdf/extracted_images/wildpaper_p1_x61_w858_h450.png",]

def main():
    if selected_model == "Qwen/Qwen3-VL-8B-Instruct":
        llm, processor = load_qwen_model(device=device)
        inputs = build_qwen_inputs(processor, query, text_context="", image_paths=image_paths, max_images=2)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = llm.generate(**inputs, max_new_tokens=512)
        response = processor.decode(outputs.sequences[0], skip_special_tokens=True)
        print("Qwen3-VL Response:")
        print(response)
        
    elif selected_model == "llava-hf/llava-v1.6-mistral-7b-hf":
        llm, processor = load_llava_model(device=device)
        inputs = build_llava_inputs(processor, query, image_paths=image_paths, max_images=2)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = llm.generate(**inputs, max_new_tokens=512)
        response = processor.decode(outputs.sequences[0], skip_special_tokens=True)
        print("LLaVA Response:")
        print(response)
    
if __name__ == "__main__":
    main()
