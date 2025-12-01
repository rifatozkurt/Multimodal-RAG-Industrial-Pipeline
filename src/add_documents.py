import os
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import documents_path, chunking_size, chunking_step, embedding_model_name, image_embedding_model_name, vectordb_path
from core.data_loaders import PdfImagesLoader, chunk_documents
from core.embedders import EmbeddingManager, EmbeddingManager_Image
from core.vectordb import VectorDBManager


def main():
    """
    This script loads documents from the documents directory, creates embeddings, and adds them to the vector DB.
    It handles both text documents from PDFs and images extracted from PDFs.
    """
    
    ## Load documents
    # text documents from pdfs
    dir_loader_pdf = DirectoryLoader(documents_path + "pdfs/",
                                        glob="*.pdf",
                                        loader_cls=PyMuPDFLoader,
                                        loader_kwargs={"extract_images": False},
                                        show_progress=True)
    documents_pdf = dir_loader_pdf.load()
    print(f"Loaded {len(documents_pdf)} text documents from PDFs.")


    # image extraction from pdfs
    os.makedirs(documents_path + "pdfs/extracted_images/", exist_ok=True)

    documents_pdf_imgs, documents_pdf_pages = PdfImagesLoader.load_from_directory(pdfs_dir=documents_path + "pdfs/",
                        images_dir=documents_path + "pdfs/extracted_images/",
                        pages_dir=documents_path + "pdfs/extracted_pages/",
                        pages_as_images=True,
                        min_dim=50,
                        min_bytes=0,
                        min_bytes_per_px=0.0,
                        extract_page_text=True,
                        store_to_folder=True,
                        page_zoom=2.0,
                        crop_pages=(2,2)
                        )
    pdf_image_paths = [doc.metadata["image_path"] for doc in documents_pdf_imgs]
    pdf_pages_paths = [doc.metadata["page_image_path"] for doc in documents_pdf_pages]
    
    # create a txt file to log extracted file paths 
    extracted_files_log = os.path.join(documents_path, "extracted_files.txt")
    if os.path.exists(extracted_files_log):
        with open(extracted_files_log, "r") as f:
            extracted_files = set(f.read().splitlines())
    else:
        extracted_files = set()

    # remove already extracted files from the documents, also for text files
    documents_pdf_imgs = [doc for doc in documents_pdf_imgs if doc.metadata["file_path"] not in extracted_files]
    documents_pdf_pages = [doc for doc in documents_pdf_pages if doc.metadata["file_path"] not in extracted_files]
    documents_pdf = [doc for doc in documents_pdf if doc.metadata["file_path"] not in extracted_files]
    
    # update the log file with newly extracted files
    filenames_to_be_logged = []
    for doc in documents_pdf_imgs + documents_pdf_pages + documents_pdf:
        if doc.metadata["file_path"] not in extracted_files and doc.metadata["file_path"] not in filenames_to_be_logged:
            filenames_to_be_logged.append(doc.metadata["file_path"])

    with open(extracted_files_log, "a") as f:
        for filename in filenames_to_be_logged:
            f.write(filename + "\n")

    print(f"Extracted {len(documents_pdf_imgs)} embedded-image documents.")
    print(f"Rendered {len(documents_pdf_pages)} page-image documents.")
    if documents_pdf_imgs:
        ex = documents_pdf_imgs[0]
        print("Example embedded image doc:", {k: ex.metadata[k] for k in ["doc_id","page","image_path","width","height"]})
    if documents_pdf_pages:
        exp = documents_pdf_pages[0]
        print("Example page image doc:", {k: exp.metadata[k] for k in ["doc_id","page","page_image_path","width","height"]})
        
    # if no new documents to process, exit
    if len(documents_pdf) == 0 and len(documents_pdf_imgs) == 0 and len(documents_pdf_pages) == 0:
        print("No new documents to process. Exiting.")
        return

    ## Create chunks
    chunks_pdf = chunk_documents(documents_pdf, chunk_size=chunking_size, chunk_step=chunking_step)
    print(f"Chunked text documents into {len(chunks_pdf)} chunks.")


    ## Create embeddings
    embedding_manager_txt = EmbeddingManager(model_name=embedding_model_name)
    embedding_manager_images = EmbeddingManager_Image(model_name=image_embedding_model_name)

    embeddings_pdf = embedding_manager_txt.create_embeddings(chunks_pdf)
    embeddings_pdf_images = embedding_manager_images.embed_images(pdf_image_paths)
    embeddings_pdf_pages = embedding_manager_images.embed_images(pdf_pages_paths)

    print(len(embeddings_pdf), len(embeddings_pdf[0]))
    print(len(embeddings_pdf_images), len(embeddings_pdf_images[0]))
    print(len(embeddings_pdf_pages), len(embeddings_pdf_pages[0]))

    # add to vector DB
    os.makedirs(vectordb_path, exist_ok=True)
        
    

    vector_db_manager_pdf = VectorDBManager(collection_name="pdf_documents_db",
                                            directory=os.path.join(vectordb_path, "pdf_db/"),
                                            source_type="pdf")
    vector_db_manager_pdf.add_documents(documents=chunks_pdf,
                                        embeddings=embeddings_pdf)

    vector_db_manager_pdf_images = VectorDBManager(collection_name="pdf_image_documents_db",
                                            directory=os.path.join(vectordb_path, "pdf_image_db/"),
                                            source_type="pdf_image")
    vector_db_manager_pdf_images.add_documents(documents=documents_pdf_imgs,
                                        embeddings=embeddings_pdf_images)
    vector_db_manager_pdf_images.add_documents(documents=documents_pdf_pages,
                                        embeddings=embeddings_pdf_pages)
    
if __name__ == "__main__":
    main()