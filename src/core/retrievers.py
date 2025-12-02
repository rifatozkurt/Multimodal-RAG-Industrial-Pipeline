# 3 different retriever classes: text-only, multimodal, and experimental multimodal. I am currently using experimental multimodal.


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


    def retrieve(self, query, query_image=None, top_k_text=3, top_k_image=3, match_threshold_text=-0.5, match_threshold_image=-0.5):

        print(f"Retrieving top: {top_k_text} text document chunks and top: {top_k_image} image document chunks...")

        try:
            query_embedding_text = self.embedding_manager_text.create_embeddings(query)[0].tolist()
            results_text = self.vector_db_text.collection.query(
                query_embeddings=[query_embedding_text],
                n_results=top_k_text
            )
            
            retrieved_doc_names_page_num = [{"doc_name": md["file_path"].split("\\")[-1], "page": md["page"]} for md in results_text["metadatas"][0] if "page" in md]

            query_embedding_image = self.embedding_manager_image.embed_texts(query_image or query)[0].tolist()
            results_image = self.vector_db_image.collection.query(
                query_embeddings=[query_embedding_image],
                #where={ "doc_id": {"$in": [doc["doc_name"] for doc in retrieved_doc_names_page_num]} },
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
        
