import uuid
import chromadb
from chromadb.config import Settings
import os

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from tqdm import tqdm
from .config import vectordb_path

class VectorDBManager:
    def __init__(self, collection_name = "unnamed", directory=vectordb_path, source_type = "undefined"):

        self.directory = directory
        self.collection_name = collection_name
        self.collection = None
        self.client = None
        self.source_type = source_type

        self._init_db()
        
    def _init_db(self):
        try:
            # create client
            os.makedirs(self.directory, exist_ok = True)
            self.client = chromadb.PersistentClient(path=self.directory)
            # create collection
            self.collection = self.client.get_or_create_collection(name = self.collection_name,
                                                                metadata = {"description" : f"Vector DB for RAG embeddings from {self.source_type}"}
                                                                )
            print(f"Vector DB initialized at {self.directory} with collection name '{self.collection_name}'")
            print(f"Currently stored vectors: {self.collection.count()}")
        except Exception as exc:
            print(f"Error initializing Vector DB: {exc}")
            raise
            
    def add_documents(self, documents, embeddings):
        """
        This function can be used to add new documents & their embeddings to the vectorDB

        Args:
            documents (list): ordered list of langchain documents
            embeddings (list): ordered list of embeddings from above documents
        """
        print(f"Adding {len(documents)} documents...")
        
        # Handle empty document list
        if len(documents) == 0:
            print("No documents to add. Skipping.")
            return
        
        # check sizes just in case
        if len(documents) != len(embeddings):
            raise ValueError(f"The number of documents ({len(documents)}) and embeddings ({len(embeddings)}) must be the same.")
        
        id_list = []
        document_content_list = []
        embedding_list = []
        metadata_list = []
        
        
        for i, (doc, embed) in enumerate(zip(documents, embeddings)):
            
            #this is just to give each entry a unique id to avoid collisions in db
            id = f"doc_{uuid.uuid4().hex[:16]}_{i}"
            id_list.append(id)
            
            metadata = doc.metadata
            metadata["doc_index"] = i
            metadata["length"] = len(doc.page_content)
            metadata["source_type"] = self.source_type
            metadata_list.append(metadata)
            
            document_content_list.append(doc.page_content)
            embedding_list.append(embed.tolist())
            
        try:
            self.collection.add(
                ids = id_list,
                metadatas = metadata_list,
                documents = document_content_list,
                embeddings = embedding_list
                )
            print(f"Successfully added {len(id_list)} documents to Vector DB.")
            print(f"Currently stored vectors: {self.collection.count()}")
                
        except Exception as exc:
            print(f"Error adding document {id} to Vector DB: {exc}")
            raise
