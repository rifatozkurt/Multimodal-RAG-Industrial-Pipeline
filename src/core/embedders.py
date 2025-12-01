import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from .config import embedding_model_name, image_embedding_model_name



class EmbeddingManager:
    """
    This class handles the embeddings of the query and the docs, using the create_embeddings() function and model_name input from HuggingFace

    ```
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """
        Attempts to load the model with model_name from SentenceTransformers Library
        """
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"EMbedding model {self.model_name} is successfully loaded. n_dim = {self.model.get_sentence_embedding_dimension()}")
        except Exception as exc:
            print(f"Error loading embedding model: {self.model_name} {exc}")
            raise
    def create_embeddings(self, documents):
        """
        Creates embeddings for the documents using the loaded model.

        Returns:
            list: List of embeddings for the documents.
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        
        if type(documents) is not str:
            texts = [doc.page_content for doc in documents]
        else:
            texts = [documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    

class EmbeddingManager_Image:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

   
    def embed_images(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            img_feats = self.model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        return img_feats.cpu().numpy()

    def embed_texts(self, texts):
        # Ensure texts are not too long for CLIP (77 token limit)
        if isinstance(texts, str):
            texts = [texts]
        
        truncated_texts = []
        for text in texts:
            if len(text) > 200:
                # Truncate and add "..." to indicate truncation
                truncated_text = text[:200].rsplit(' ', 1)[0] + "..."
                print(f"Warning: Truncated long text for CLIP embedding: {text[:50]}...")
                truncated_texts.append(truncated_text)
            else:
                truncated_texts.append(text)
        
        inputs = self.processor(text=truncated_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            txt_feats = self.model.get_text_features(**inputs)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        return txt_feats.cpu().numpy()