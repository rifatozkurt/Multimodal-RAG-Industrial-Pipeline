import numpy as np
from pathlib import Path

# 3 different retriever classes: text-only, multimodal, and experimental multimodal. I am currently using experimental multimodal and retriever multi modal image page text. 


def merge_embeddings(q, results_clip, merge_type="normalized_mean", *, alpha=10.0, lam=0.3, eps=1e-12):

    try:
        emb_list = results_clip.get("embeddings", [None])[0]
    except Exception:
        return None

    if emb_list is None:
        return None
    try:
        if len(emb_list) == 0:
            return None
    except TypeError:
        return None

    T = np.asarray(emb_list, dtype=np.float32)  # (K, D)
    if T.ndim != 2 or T.shape[0] == 0:
        return None

    qv = np.asarray(q, dtype=np.float32).reshape(-1)  # (D,)
    if qv.ndim != 1 or qv.shape[0] != T.shape[1]:
        return None

    def l2_normalize(x):
        n = np.linalg.norm(x)
        if n < eps:
            return x
        return x / n

    # unit vectors
    qn = l2_normalize(qv)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + eps)

    merge_type = (merge_type or "").lower()

    # 1) Normalized mean
    if merge_type == "normalized_mean":
        m = Tn.mean(axis=0)
        m = l2_normalize(m)
        return m.tolist()

    # 2) Similarity-weighted merge (softmax)
    if merge_type == "similarity_weighted":
        sims = Tn @ qn  # (K,) cosine similarities
        # Softmax(alpha * sims)
        x = alpha * sims
        x = x - np.max(x)  # for stability
        w = np.exp(x)
        w = w / (np.sum(w) + eps)  
        m = (w[:, None] * Tn).sum(axis=0)
        m = l2_normalize(m)
        return m.tolist()

    # 3) Query interpolation: m = lam*q + (1-lam)*context_merge
    if merge_type == "query_interpolation":
        # Use a strong context merge by default: similarity-weighted
        sims = Tn @ qn
        x = alpha * sims
        x = x - np.max(x)
        w = np.exp(x)
        w = w / (np.sum(w) + eps)
        context = (w[:, None] * Tn).sum(axis=0)
        context = l2_normalize(context)

        m = lam * qn + (1.0 - lam) * context
        m = l2_normalize(m)
        return m.tolist()

    # 4) PCA principal direction (signed toward query)
    if merge_type == "pca":
        # Center on mean direction (on the sphere; mean then normalize)
        mu = Tn.mean(axis=0)
        mu = l2_normalize(mu)

        X = Tn - mu[None, :] 

        # If K < 2, PCA is ill-defined, back to normalized mean
        if X.shape[0] < 2:
            m = mu
            return m.tolist()

        # Compute first principal component via SVD
        # X = U S Vt, pc1 = Vt[0]
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            pc1 = Vt[0].astype(np.float32)  
        except Exception:
            m = mu
            return m.tolist()

        pc1 = l2_normalize(pc1)

        # Choose sign so that it aligns with the query direction
        if np.dot(pc1, qn) < 0:
            pc1 = -pc1

        # Build merged vector: mean direction plus a small move along pc1
        # The step size can be 1.0; normalization will keep it stable.
        m = mu + pc1
        m = l2_normalize(m)
        return m.tolist()

    raise ValueError(
        f"Unknown merge_type='{merge_type}'. "
        "Use one of: normalized_mean, similarity_weighted, query_interpolation, pca."
    )



class RetrieverText:
    def __init__(self, vector_db, embedding_manager):
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager

        
    def retrieve(self, query, top_k, match_threshold=-0.5):
        
        print(f"Retrieving {top_k} document chunks...")
        
        try:
            query_embedding = self.embedding_manager.create_embeddings(query)[0].tolist()
            results = self.vector_db.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            results_list = []
            
            if results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    score = 1 - results["distances"][0][i]  
                    if score >= match_threshold:     # cosine distance <= 1.5 seems reasonable here
                        result = {
                            "id": results["ids"][0][i],
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "modality": "text",
                            "score": score
                        }
                        results_list.append(result)
                print(f"Retrieved {len(results_list)} documents above the threshold of {match_threshold}.")
            else:
                print("No documents retrieved.")
            
                
            
            return results_list            
            
        except Exception as exc:
            print(f"Error retrieving: {exc}")
            raise
    
    


class RetrieverMultiModal:
    def __init__(self, vector_db_text, vector_db_image, embedding_manager_text, embedding_manager_image):
        self.vector_db_text = vector_db_text
        self.vector_db_image = vector_db_image
        self.embedding_manager_text = embedding_manager_text
        self.embedding_manager_image = embedding_manager_image


    def retrieve(self, query, top_k_text, top_k_image, match_threshold_text=-0.5, match_threshold_image=-0.5):

        print(f"Retrieving {top_k_text} text document chunks and {top_k_image} image document chunks...")

        try:
            query_embedding_text = self.embedding_manager_text.create_embeddings(query)[0].tolist()
            results_text = self.vector_db_text.collection.query(
                query_embeddings=[query_embedding_text],
                n_results=top_k_text
            )

            query_embedding_image = self.embedding_manager_image.embed_texts(query)[0].tolist()
            results_image = self.vector_db_image.collection.query(
                query_embeddings=[query_embedding_image],
                n_results=top_k_image
            )
        
            results_list_text = []

            if results_text["ids"][0]:
                for i in range(len(results_text["ids"][0])):
                    score = 1 - results_text["distances"][0][i]
                    if score >= match_threshold_text:  # cosine distance <= 1.5 seems reasonable here
                        result = {
                            "id": results_text["ids"][0][i],
                            "document": results_text["documents"][0][i],
                            "metadata": results_text["metadatas"][0][i],
                            "modality": "text",
                            "score": score
                        }
                        results_list_text.append(result)
                print(f"Retrieved {len(results_list_text)} documents above the threshold of {match_threshold_text}.")
            else:
                print("No text documents retrieved.")
            
            results_list_image = []
            if results_image["ids"][0]:
                for i in range(len(results_image["ids"][0])):
                    score = 1 - results_image["distances"][0][i]
                    if score >= match_threshold_image:  # cosine distance <= 1.5 seems reasonable here
                        result = {
                            "id": results_image["ids"][0][i],
                            "document": results_image["documents"][0][i],
                            "metadata": results_image["metadatas"][0][i],
                            "modality": "image",
                            "score": score
                        }
                        results_list_image.append(result)
                print(f"Retrieved {len(results_list_image)} image documents above the threshold of {match_threshold_image}.")
            else:
                print("No image documents retrieved.")

            return results_list_text, results_list_image

        except Exception as exc:
            print(f"Error retrieving: {exc}")
            raise
        

class RetrieverMultiModal_experimental:
    def __init__(self, vector_db_text, vector_db_image, embedding_manager_text, embedding_manager_image):
        self.vector_db_text = vector_db_text
        self.vector_db_image = vector_db_image
        self.embedding_manager_text = embedding_manager_text
        self.embedding_manager_image = embedding_manager_image


    def retrieve(
        self,
        query,
        query_image=None,
        top_k_text=3,
        top_k_image=3,
        match_threshold_text=-0.5,
        match_threshold_image=-0.5,
        filtered_image_retrieval=True,
    ):

        print(f"Retrieving top: {top_k_text} text document chunks and top: {top_k_image} image document chunks...")

        try:
            query_embedding_text = self.embedding_manager_text.create_embeddings(query)[0].tolist()
            results_text = self.vector_db_text.collection.query(
                query_embeddings=[query_embedding_text],
                n_results=top_k_text
            )
            
            results_list_text = []

            if results_text["ids"][0]:
                for i in range(len(results_text["ids"][0])):
                    score = 1 - results_text["distances"][0][i]
                    if score >= match_threshold_text:  # cosine distance <= 1.5 seems reasonable here
                        result = {
                            "id": results_text["ids"][0][i],
                            "document": results_text["documents"][0][i],
                            "metadata": results_text["metadatas"][0][i],
                            "modality": "text",
                            "score": score
                        }
                        results_list_text.append(result)
                print(f"Retrieved {len(results_list_text)} documents above the threshold of {match_threshold_text}.")
            else:
                print("No text documents retrieved.")
            
            query_embedding_image = self.embedding_manager_image.embed_texts(query_image or query)[0].tolist()
            results_image = None
            if filtered_image_retrieval:
                allowed_file_paths = {
                    result["metadata"].get("file_path")
                    for result in results_list_text
                    if result.get("metadata", {}).get("file_path")
                }
                allowed_doc_ids = {
                    result["metadata"].get("doc_id")
                    for result in results_list_text
                    if result.get("metadata", {}).get("doc_id")
                }

                where_clause = None
                if allowed_file_paths:
                    where_clause = {"file_path": {"$in": sorted(allowed_file_paths)}}
                elif allowed_doc_ids:
                    where_clause = {"doc_id": {"$in": sorted(allowed_doc_ids)}}

                if where_clause:
                    results_image = self.vector_db_image.collection.query(
                        query_embeddings=[query_embedding_image],
                        where=where_clause,
                        n_results=top_k_image,
                    )
                else:
                    print("No matching text documents found for filtered image retrieval.")
                    results_image = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            else:
                results_image = self.vector_db_image.collection.query(
                    query_embeddings=[query_embedding_image],
                    n_results=top_k_image,
                )

            results_list_image = []
            if results_image["ids"][0]:
                for i in range(len(results_image["ids"][0])):
                    score = 1 - results_image["distances"][0][i]
                    if score >= match_threshold_image:  # cosine distance <= 1.5 seems reasonable here
                        result = {
                            "id": results_image["ids"][0][i],
                            "document": results_image["documents"][0][i],
                            "metadata": results_image["metadatas"][0][i],
                            "modality": "image",
                            "score": score
                        }
                        results_list_image.append(result)
                print(f"Retrieved {len(results_list_image)} image documents above the threshold of {match_threshold_image}.")
            else:
                print("No image documents retrieved.")

            return results_list_text, results_list_image

        except Exception as exc:
            print(f"Error retrieving: {exc}")
            raise


class RetrieverMultiModal_ImagePageText:
    def __init__(self, vector_db_text, vector_db_image_text, embedding_manager_text):
        self.vector_db_text = vector_db_text
        self.vector_db_image_text = vector_db_image_text
        self.embedding_manager_text = embedding_manager_text

    def retrieve(
        self,
        query,
        query_image=None,
        top_k_text=3,
        top_k_image=3,
        match_threshold_text=-0.5,
        match_threshold_image=-0.5,
        filtered_image_retrieval=True,
    ):
        print(
            f"Retrieving top: {top_k_text} text document chunks and top: {top_k_image} image-page-text chunks..."
        )

        try:
            query_embedding_text = self.embedding_manager_text.create_embeddings(query)[0].tolist()
            results_text = self.vector_db_text.collection.query(
                query_embeddings=[query_embedding_text],
                n_results=top_k_text,
            )

            results_list_text = []
            if results_text["ids"][0]:
                for i in range(len(results_text["ids"][0])):
                    score = 1 - results_text["distances"][0][i]
                    if score >= match_threshold_text:
                        result = {
                            "id": results_text["ids"][0][i],
                            "document": results_text["documents"][0][i],
                            "metadata": results_text["metadatas"][0][i],
                            "modality": "text",
                            "score": score,
                        }
                        results_list_text.append(result)
                print(
                    f"Retrieved {len(results_list_text)} documents above the threshold of {match_threshold_text}."
                )
            else:
                print("No text documents retrieved.")

            image_query = query_image or query
            query_embedding_image = self.embedding_manager_text.create_embeddings(image_query)[0].tolist()
            results_image = None
            if filtered_image_retrieval:
                allowed_file_paths = {
                    result["metadata"].get("file_path")
                    for result in results_list_text
                    if result.get("metadata", {}).get("file_path")
                }
                allowed_doc_ids = {
                    result["metadata"].get("doc_id")
                    for result in results_list_text
                    if result.get("metadata", {}).get("doc_id")
                }

                where_clause = None
                if allowed_file_paths:
                    where_clause = {"file_path": {"$in": sorted(allowed_file_paths)}}
                elif allowed_doc_ids:
                    where_clause = {"doc_id": {"$in": sorted(allowed_doc_ids)}}

                if where_clause:
                    results_image = self.vector_db_image_text.collection.query(
                        query_embeddings=[query_embedding_image],
                        where=where_clause,
                        n_results=top_k_image,
                    )
                else:
                    print("No matching text documents found for filtered image retrieval.")
                    results_image = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            else:
                results_image = self.vector_db_image_text.collection.query(
                    query_embeddings=[query_embedding_image],
                    n_results=top_k_image,
                )

            results_list_image = []
            if results_image["ids"][0]:
                for i in range(len(results_image["ids"][0])):
                    score = 1 - results_image["distances"][0][i]
                    if score >= match_threshold_image:
                        result = {
                            "id": results_image["ids"][0][i],
                            "document": results_image["documents"][0][i],
                            "metadata": results_image["metadatas"][0][i],
                            "modality": "image",
                            "score": score,
                        }
                        results_list_image.append(result)
                print(
                    f"Retrieved {len(results_list_image)} image-page-text documents above the threshold of {match_threshold_image}."
                )
            else:
                print("No image-page-text documents retrieved.")

            return results_list_text, results_list_image

        except Exception as exc:
            print(f"Error retrieving: {exc}")
            raise


class RetrieverMultiModal_ImageVLMCaptions:
    def __init__(self, vector_db_text, vector_db_image_captions, embedding_manager_text):
        self.vector_db_text = vector_db_text
        self.vector_db_image_captions = vector_db_image_captions
        self.embedding_manager_text = embedding_manager_text

    @staticmethod
    def _doc_id_candidates(doc_id: str | None, file_path: str | None) -> set[str]:
        candidates: set[str] = set()

        def add_candidate(value: str | None):
            if not value:
                return
            value = str(value).strip()
            if not value:
                return
            candidates.add(value)
            candidates.add(value.lower())
            candidates.add(value.upper())

        add_candidate(doc_id)

        if file_path:
            try:
                file_name = Path(str(file_path)).name
                add_candidate(file_name)
                stem = Path(file_name).stem
                add_candidate(stem)
            except Exception:
                pass

        # Strip a trailing .pdf extension if present.
        stripped: set[str] = set()
        for value in list(candidates):
            if value.lower().endswith(".pdf"):
                stripped.add(value[:-4])
        for value in stripped:
            add_candidate(value)

        return {c for c in candidates if c}

    @staticmethod
    def _caption_id_candidates(metadata: dict) -> set[str]:
        doc_id = metadata.get("document_id") or metadata.get("doc_id")
        file_path = (
            metadata.get("image_path")
            or metadata.get("image_rel_path")
            or metadata.get("filename")
        )
        return RetrieverMultiModal_ImageVLMCaptions._doc_id_candidates(doc_id, file_path)

    def retrieve(
        self,
        query,
        query_image=None,
        top_k_text=3,
        top_k_image=3,
        match_threshold_text=-0.5,
        match_threshold_image=-0.5,
        filtered_image_retrieval=True,
    ):
        print(
            f"Retrieving top: {top_k_text} text document chunks and top: {top_k_image} image-caption chunks..."
        )

        try:
            query_embedding_text = self.embedding_manager_text.create_embeddings(query)[0].tolist()
            results_text = self.vector_db_text.collection.query(
                query_embeddings=[query_embedding_text],
                n_results=top_k_text,
            )

            results_list_text = []
            if results_text["ids"][0]:
                for i in range(len(results_text["ids"][0])):
                    score = 1 - results_text["distances"][0][i]
                    if score >= match_threshold_text:
                        result = {
                            "id": results_text["ids"][0][i],
                            "document": results_text["documents"][0][i],
                            "metadata": results_text["metadatas"][0][i],
                            "modality": "text",
                            "score": score,
                        }
                        results_list_text.append(result)
                print(
                    f"Retrieved {len(results_list_text)} documents above the threshold of {match_threshold_text}."
                )
            else:
                print("No text documents retrieved.")

            image_query = query_image or query
            query_embedding_image = self.embedding_manager_text.create_embeddings(image_query)[0].tolist()
            results_image = None
            if filtered_image_retrieval:
                allowed_doc_ids: set[str] = set()
                for result in results_list_text:
                    metadata = result.get("metadata", {}) or {}
                    doc_id = metadata.get("doc_id")
                    file_path = metadata.get("file_path") or metadata.get("source")
                    allowed_doc_ids.update(self._doc_id_candidates(doc_id, file_path))

                where_clause = None
                if allowed_doc_ids:
                    where_clause = {"document_id": {"$in": sorted(allowed_doc_ids)}}

                if where_clause:
                    results_image = self.vector_db_image_captions.collection.query(
                        query_embeddings=[query_embedding_image],
                        where=where_clause,
                        n_results=top_k_image,
                    )
                    if not results_image["ids"][0]:
                        print(
                            "No matching VLM captions found for filtered document IDs; "
                            "falling back to unfiltered retrieval and filtering locally."
                        )
                        results_image = self.vector_db_image_captions.collection.query(
                            query_embeddings=[query_embedding_image],
                            n_results=max(top_k_image * 10, top_k_image),
                        )
                else:
                    print("No matching text documents found for filtered image retrieval.")
                    results_image = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            else:
                results_image = self.vector_db_image_captions.collection.query(
                    query_embeddings=[query_embedding_image],
                    n_results=top_k_image,
                )

            results_list_image = []
            if results_image["ids"][0]:
                for i in range(len(results_image["ids"][0])):
                    score = 1 - results_image["distances"][0][i]
                    if filtered_image_retrieval and allowed_doc_ids:
                        metadata = results_image["metadatas"][0][i] or {}
                        if not (self._caption_id_candidates(metadata) & allowed_doc_ids):
                            continue
                    if score >= match_threshold_image:
                        result = {
                            "id": results_image["ids"][0][i],
                            "document": results_image["documents"][0][i],
                            "metadata": results_image["metadatas"][0][i],
                            "modality": "image",
                            "score": score,
                        }
                        results_list_image.append(result)
                print(
                    f"Retrieved {len(results_list_image)} image-caption documents above the threshold of {match_threshold_image}."
                )
            else:
                print("No image-caption documents retrieved.")

            return results_list_text, results_list_image

        except Exception as exc:
            print(f"Error retrieving: {exc}")
            raise


class RetrieverMultiModal_ImageCLIPRepresentations:
    def __init__(
        self,
        vector_db_text,
        vector_db_text_clip,
        vector_db_image,
        embedding_manager_text,
        embedding_manager_text_clip,
    ):
        self.vector_db_text = vector_db_text
        self.vector_db_text_clip = vector_db_text_clip
        self.vector_db_image = vector_db_image
        self.embedding_manager_text = embedding_manager_text
        self.embedding_manager_text_clip = embedding_manager_text_clip

    def retrieve(
        self,
        query,
        query_image=None,
        top_k_text=3,
        top_k_image=3,
        clip_top_k=10,
        match_threshold_text=-0.5,
        match_threshold_image=-0.5,
        filtered_image_retrieval=True,
        merge_type="normalized_mean",
    ):
        print(
            f"Retrieving top: {top_k_text} text chunks, then top: {clip_top_k} CLIP-text chunks, "
            f"and top: {top_k_image} images..."
        )

        try:
            query_embedding_text = self.embedding_manager_text.create_embeddings(query)[0].tolist()
            results_text = self.vector_db_text.collection.query(
                query_embeddings=[query_embedding_text],
                n_results=top_k_text,
            )

            results_list_text = []
            if results_text["ids"][0]:
                for i in range(len(results_text["ids"][0])):
                    score = 1 - results_text["distances"][0][i]
                    if score >= match_threshold_text:
                        result = {
                            "id": results_text["ids"][0][i],
                            "document": results_text["documents"][0][i],
                            "metadata": results_text["metadatas"][0][i],
                            "modality": "text",
                            "score": score,
                        }
                        results_list_text.append(result)
                print(
                    f"Retrieved {len(results_list_text)} documents above the threshold of {match_threshold_text}."
                )
            else:
                print("No text documents retrieved.")

            image_query = query_image or query
            query_embedding_clip = (
                self.embedding_manager_text_clip.create_embeddings(image_query)[0].tolist()
            )
            results_clip = self.vector_db_text_clip.collection.query(
                query_embeddings=[query_embedding_clip],
                n_results=clip_top_k,
                include=["embeddings", "documents", "metadatas", "distances"],
            )

            merged_embedding = None
            clip_embeddings = results_clip.get("embeddings", [None])[0]
            if clip_embeddings is not None and len(clip_embeddings) > 0:
                merged_embedding = merge_embeddings(
                    query_embedding_clip,
                    results_clip,
                    merge_type=merge_type,
                )

            if merged_embedding is None:
                print("No CLIP text embeddings found to merge. Skipping image retrieval.")
                results_image = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            else:
                if filtered_image_retrieval:
                    allowed_file_paths = {
                        result["metadata"].get("file_path")
                        for result in results_list_text
                        if result.get("metadata", {}).get("file_path")
                    }
                    allowed_doc_ids = {
                        result["metadata"].get("doc_id")
                        for result in results_list_text
                        if result.get("metadata", {}).get("doc_id")
                    }

                    where_clause = None
                    if allowed_file_paths:
                        where_clause = {"file_path": {"$in": sorted(allowed_file_paths)}}
                    elif allowed_doc_ids:
                        where_clause = {"doc_id": {"$in": sorted(allowed_doc_ids)}}

                    if where_clause:
                        results_image = self.vector_db_image.collection.query(
                            query_embeddings=[merged_embedding],
                            where=where_clause,
                            n_results=top_k_image,
                        )
                    else:
                        print("No matching text documents found for filtered image retrieval.")
                        results_image = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
                else:
                    results_image = self.vector_db_image.collection.query(
                        query_embeddings=[merged_embedding],
                        n_results=top_k_image,
                    )

            results_list_image = []
            if results_image["ids"][0]:
                for i in range(len(results_image["ids"][0])):
                    score = 1 - results_image["distances"][0][i]
                    if score >= match_threshold_image:
                        result = {
                            "id": results_image["ids"][0][i],
                            "document": results_image["documents"][0][i],
                            "metadata": results_image["metadatas"][0][i],
                            "modality": "image",
                            "score": score,
                        }
                        results_list_image.append(result)
                print(
                    f"Retrieved {len(results_list_image)} image documents above the threshold of {match_threshold_image}."
                )
            else:
                print("No image documents retrieved.")

            return results_list_text, results_list_image

        except Exception as exc:
            print(f"Error retrieving: {exc}")
            raise
        
