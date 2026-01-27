# 3 different retriever classes: text-only, multimodal, and experimental multimodal. I am currently using experimental multimodal and retriever multi modal image page text. 


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
                allowed_doc_ids = {
                    result["metadata"].get("doc_id")
                    for result in results_list_text
                    if result.get("metadata", {}).get("doc_id")
                }

                where_clause = None
                if allowed_doc_ids:
                    where_clause = {"document_id": {"$in": sorted(allowed_doc_ids)}}

                if where_clause:
                    results_image = self.vector_db_image_captions.collection.query(
                        query_embeddings=[query_embedding_image],
                        where=where_clause,
                        n_results=top_k_image,
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
        
