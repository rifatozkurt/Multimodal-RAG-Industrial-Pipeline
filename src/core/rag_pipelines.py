from .retrievers import RetrieverMultiModal, RetrieverText, RetrieverMultiModal_experimental
from .models_llm import *
import torch

class SimpleRAG:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.message_history = []
        
    def generate_response(self, query, top_k=3, match_threshold=-0.5):
        
        print(f"Generating response for query: {query}")
        try:
            retrieved_docs = self.retriever.retrieve(query=query, top_k=top_k, match_threshold=match_threshold)
            
            # I test only with my documents, so I do not want generation without context. You can remove this return to continue generation without context.        
            if not retrieved_docs:
                return ("No relevant documents found.")
            
            context = "\n\n".join([doc["document"] for doc in retrieved_docs])
            print(f"Context for LLM:\n{context[:500]}...\n")
            
            messages = message_general(query=query, context=context)
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            if answer:
                print(f"Answer received from LLM.")
                
            # add to message history
            self.message_history.append({
                "query": query,
                "context": context,
                "answer": answer
            })
            
            return answer
        
        except Exception as exc:
            print(f"Error generating response: {exc}")
            raise
        
        
        
class AdvancedRAG:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.message_history = []
        
    def generate_response(self, query, top_k=3, match_threshold=-0.5, preprocess_type=None, chunk_size=1000, summarize=False):
        
        print(f"Generating response for query: {query}")
        try:
            
            # Preprocessing the query into an expanded or chain-of-thought format for better retrieval
            if preprocess_type:
                if preprocess_type == "expand":
                    expander_query = message_expander(query=query, length=chunk_size)
                    response = self.llm.invoke(expander_query)
                    query_retrieval = response.content
                    print(f"Expanded retrieval query: {query_retrieval}")
                if preprocess_type == "chain_of_thought":
                    cot_query = message_cot(query=query)
                    response = self.llm.invoke(cot_query)
                    query_retrieval = response.content
                    print(f"Chain of thought retrieval query: {query_retrieval}")
            else:
                query_retrieval = query
            
            retrieved_docs = self.retriever.retrieve(query=query_retrieval, top_k=top_k, match_threshold=match_threshold)
            
            # I test only with my documents, so I do not want generation without context. You can remove this return to continue generation without context.
            if not retrieved_docs:
                return {"answer": "No relevant documents found. Stopping to avoid hallucinations.", "sources": None, "retrieved_docs": None, "message_history": self.message_history}
            
            context = "\n\n".join([doc["document"] for doc in retrieved_docs])
            sources = [{"source": doc["metadata"].get("source", "unknown"), "score": doc["score"]} for doc in retrieved_docs]
            
            print(f"Context for LLM:\n{context[:500]}...\n")
            
            messages = message_general(query=query, context=context)
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            
            if answer:
                print(f"Answer received from LLM.")
                
            if summarize:
                # Summarization step
                summarization_prompt = message_summarizer(message=answer)
                summary_response = self.llm.invoke(summarization_prompt)
                answer = summary_response.content
                print(f"Summarized Answer received from LLM.")
                
            # add to message history
            self.message_history.append({
                "query": query,
                "context": context,
                "answer": answer
            })
            
            return {"answer": answer, "sources": sources, "retrieved_docs": retrieved_docs, "message_history": self.message_history}
        
        except Exception as exc:
            print(f"Error generating response: {exc}")
            raise
        
        

class AdvancedMultimodalRAG:
    """
    Advanced multimodal RAG for image + text retrieval and generation
    using either LLaVA or Qwen3-VL as the multimodal model.

    - Uses RetrieverMultiModal class to get:
        * text chunks from your text vector DB 
        * image docs (paths in metadata) from your image vector DB
    - Extracts the page texts from pages where images were found
    - Builds a text context from retrieved text + page_text from images
    - Selects top-k images and passes them to the chosen backend
    - Optional: query expansion / chain-of-thought via a text LLM
    - Optional: summarization of the final answer
    
    - future: hierarchical retrieval (crop pages, then retrieve larger crops or full pages based on initial retrieval)
    - future: cot retrieval query for image retrieval separately, in the form of image descriptions?
    """

    def __init__(
        self,
        retriever_multimodal,
        model_llava=None,
        processor_llava=None,
        model_qwen=None,
        processor_qwen=None,
        text_llm=None):         # optional: Groq text LLM for query expansion / summarization
    
        self.retriever_multimodal = retriever_multimodal

        self.model_llava = model_llava
        self.processor_llava = processor_llava

        self.model_qwen = model_qwen
        self.processor_qwen = processor_qwen

        self.text_llm = text_llm
        self.message_history = []

    def _preprocess_query(self, query, preprocess_type=None, chunk_size=1000):
        """
        Optional query preprocessing using a text LLM (Groq), if provided.

        preprocess_type:
            - None
            - "expand"
            - "chain_of_thought"
            - future: cot for image retrieval in caption style
        """
        if not preprocess_type or self.text_llm is None:
            return query

        if preprocess_type == "expand":
            expander_query = message_expander(query=query, length=chunk_size)
            response = self.text_llm.invoke(expander_query)
            query_retrieval = response.content
            print("Expanded retrieval query:\n", query_retrieval[:500], "...\n")
            return query_retrieval

        if preprocess_type == "chain_of_thought":
            cot_query = message_cot(query=query)
            response = self.text_llm.invoke(cot_query)
            query_retrieval = response.content
            print("Chain-of-thought retrieval query:\n", query_retrieval[:500], "...\n")
            return query_retrieval

        # Fallback to normal query if not selected
        return query
    
    def _preprocess_image_query(self, query):
        """
        Image query preprocessing for CLIP compatibility.
        - If text LLM is available: Generate a concise image caption
        - If no text LLM: Apply CLIP-safe truncation to original query
        Returns a query suitable for CLIP embedding (≤200 characters).
        """
        if self.text_llm is None:
            # No text LLM available - apply CLIP-safe truncation
            if len(query) > 200:
                truncated_query = query[:200].rsplit(' ', 1)[0] + "..."
                print(f"No text LLM available. Truncated query for CLIP: {truncated_query}")
                return truncated_query
            else:
                return query

        # Generate image caption using text LLM
        caption_query = message_image_caption_generator(message=query)
        response = self.text_llm.invoke(caption_query)
        image_caption = response.content
        
        # Ensure caption is not too long for CLIP (additional safety)
        if len(image_caption) > 200:
            # Take first sentence if caption is too long
            first_sentence = image_caption.split('.')[0] + '.'
            image_caption = first_sentence if len(first_sentence) <= 200 else image_caption[:200]
            print(f"Truncated long image caption for CLIP compatibility.")
        
        print("Generated image caption for retrieval:\n", image_caption, "\n")
        return image_caption

    def _build_text_context(self, text_docs, image_docs, max_text_chunks):
        """
        Build a structured text context:
        - Section 1: Retrieved text chunks
        - Section 2: Page text extracted from image locations
        """

        sections = []

        # ------------------------------------------
        # Retrieved text chunks
        text_parts = []
        for i, doc in enumerate(text_docs[:max_text_chunks]):
            chunk_label = f"[Text Chunk {i+1}]"
            chunk_body = doc["document"]
            if chunk_body:
                text_parts.append(f"{chunk_label}\n{chunk_body}")

        if text_parts:
            section_text = "### Retrieved Text Chunks\n" + "\n\n".join(text_parts)
            sections.append(section_text)

        # ------------------------------------------
        # Page text from images
        image_text_parts = []
        for j, img_doc in enumerate(image_docs):
            page_text = img_doc["metadata"].get("page_text")
            if page_text:
                img_label = f"[Image Context {j+1}]"
                image_text_parts.append(f"{img_label}\n{page_text}")

        if image_text_parts:
            section_page_text = "### Page Text Near Retrieved Images\n" + "\n\n".join(image_text_parts)
            sections.append(section_page_text)

        # if no context found
        if not sections:
            return ""

        # Combine all sections
        final_context = "\n\n====================\n\n".join(sections)
        return final_context


    def _select_image_paths(self, image_docs, max_images):
        """
        Collect image paths from image_docs' metadata.
        Handles both 'image_path' and 'page_image_path'.
        """
        paths = []
        for img_doc in image_docs:
            p = img_doc["metadata"].get("image_path") or img_doc["metadata"].get("page_image_path")
            if p:
                paths.append(p)
            if len(paths) >= max_images:
                break
        return paths

    def _run_llava(self, query, text_context, image_paths, max_images, max_new_tokens, device):
        
        if self.model_llava is None or self.processor_llava is None:
            raise ValueError("LLaVA model/processor not provided to AdvancedMultimodalRAG.")

        inputs = build_llava_inputs(
            processor_llava=self.processor_llava,
            query=query,
            text_context=text_context,
            image_paths=image_paths,
            max_images=max_images,
            device=device)

        with torch.no_grad():
            output_ids = self.model_llava.generate(
                **inputs,
                max_new_tokens=max_new_tokens)

        # LlavaNextProcessor supports decode
        answer = self.processor_llava.decode(output_ids[0], skip_special_tokens=True)
        return answer

    def _run_qwen(
        self,
        query,
        text_context,
        image_paths,
        max_images,
        max_new_tokens):
        if self.model_qwen is None or self.processor_qwen is None:
            raise ValueError("Qwen3-VL model/processor not provided to AdvancedMultimodalRAG.")

        inputs = build_qwen_inputs(
            processor_qwen=self.processor_qwen,
            query=query,
            text_context=text_context,
            image_paths=image_paths,
            max_images=max_images)

        inputs = inputs.to(self.model_qwen.device)

        with torch.no_grad():
            generated_ids = self.model_qwen.generate(
                **inputs,
                max_new_tokens=max_new_tokens)

        # Trim prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer_list = self.processor_qwen.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # single example => take first
        answer = answer_list[0] if len(answer_list) > 0 else ""
        return answer

    def generate_response(
        self,
        query,
        backend="qwen_vl",     # "qwen_vl" or "llava"
        top_k_text=3,
        top_k_image=3,
        match_threshold_text=-0.5,
        match_threshold_image=-0.5,
        max_images=1,
        max_new_tokens=128,
        preprocess_type=None,  # None, "expand", "chain_of_thought"
        chunk_size=1000,
        summarize=False,
        image_query_captioning=False,
        device="cuda:0"):
        """
        Main entry point.

        backend:
            - "qwen_vl" : Qwen3-VL-8B
            - "llava"   : LLaVA v1.6 (Mistral 7B)

        Returns a dict with:
            - "answer"
            - "backend"
            - "query_original"
            - "query_retrieval"
            - "text_context"
            - "image_paths_used"
            - "retrieved_text_docs"
            - "retrieved_image_docs"
            - "sources_text"
            - "sources_image"
            - "message_history"
        """

        print("Generating multimodal response for query:", query)
        # 1) optional query preprocessing
        query_retrieval = self._preprocess_query(
            query=query,
            preprocess_type=preprocess_type,
            chunk_size=chunk_size)
        
        # Handle image query with CLIP token limit safety
        if image_query_captioning:
            # User explicitly wants image captioning
            query_image_retrieval = self._preprocess_image_query(query=query)
        elif len(query) > 200:
            # Query is too long for CLIP - automatically generates a caption
            print(f"Query too long for CLIP ({len(query)} chars). Auto-generating image caption...")
            query_image_retrieval = self._preprocess_image_query(query=query)
        else:
            # Query is short enough - use original query for image retrieval
            query_image_retrieval = query
        
        # 2) retrieval
        try:
            text_docs, image_docs = self.retriever_multimodal.retrieve(
                query=query_retrieval,
                query_image=query_image_retrieval,
                top_k_text=top_k_text,
                top_k_image=top_k_image,
                match_threshold_text=match_threshold_text,
                match_threshold_image=match_threshold_image)
        except TypeError as e:
            # Handle case where retriever doesn't support query_image parameter
            if "query_image" in str(e):
                text_docs, image_docs = self.retriever_multimodal.retrieve(
                    query=query_retrieval,
                    top_k_text=top_k_text,
                    top_k_image=top_k_image,
                    match_threshold_text=match_threshold_text,
                    match_threshold_image=match_threshold_image)
            else:
                raise
        except Exception as exc:
            print("Error during retrieval:", exc)
            raise

        if not text_docs and not image_docs:
            print("No relevant multimodal documents found.")
            """
            result = {
                "answer": "No relevant documents found. Stopping to avoid hallucinations.",
                "backend": backend,
                "query_original": query,
                "query_retrieval": query_retrieval,
                "text_context": "",
                "image_paths_used": [],
                "retrieved_text_docs": [],
                "retrieved_image_docs": [],
                "sources_text": [],
                "sources_image": [],
                "message_history": self.message_history
            }
            self.message_history.append(result)
            return result
            """

        # 3) build text context
        text_context = self._build_text_context(
            text_docs=text_docs,
            image_docs=image_docs,
            max_text_chunks=top_k_text)
        print("Text context length:", len(text_context))

        # 4) image selection
        image_paths_used = self._select_image_paths(
            image_docs=image_docs,
            max_images=max_images)
        print("Using", len(image_paths_used), "image(s) for generation.")

        # 5) run the selected backend
        if backend == "llava":
            answer = self._run_llava(
                query=query,
                text_context=text_context,
                image_paths=image_paths_used,
                max_images=max_images,
                max_new_tokens=max_new_tokens,
                device=device)
            
        elif backend == "qwen_vl":
            answer = self._run_qwen(
                query=query,
                text_context=text_context,
                image_paths=image_paths_used,
                max_images=max_images,
                max_new_tokens=max_new_tokens)
        else:
            raise ValueError("Unknown backend: {} (use 'llava' or 'qwen_vl')".format(backend))

        print("Answer from backend {}: \n{}\n".format(backend, answer[:500]))

        # 6) optional summarization using text LLM
        if summarize and self.text_llm is not None:
            print("Summarizing answer with text LLM...")
            summarization_prompt = message_summarizer(message=answer)
            summary_response = self.text_llm.invoke(summarization_prompt)
            answer = summary_response.content
            print("Summarized answer:\n", answer[:500], "...\n")

        # 7) collect sources
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

        # 8) build result
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
        }

        self.message_history.append(result)
        return result
