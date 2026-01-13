######### General Configurations #########
documents_path = "./documents/" # relative path of all doc directories
vectordb_path = documents_path + "vectorDB/"
chunking_size = 1000  # Size of each chunk
chunking_step = 200 # Step size btw chunks

embedding_model_name = 'all-MiniLM-L6-v2' #'all-mpnet-base-v2' #'all-MiniLM-L6-v2'  # SentenceTransformer model name
image_embedding_model_name = "openai/clip-vit-large-patch14"  # CLIP model name

eval_dataset_path = "./eval/datasets/dataset_mcq_fib.json"  # path to evaluation dataset
eval_runs_path = "./eval/runs/"  # path to save evaluation results

# Evaluation model defaults
nli_model_name = "cross-encoder/nli-deberta-v3-small"
