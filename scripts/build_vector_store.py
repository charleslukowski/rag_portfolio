import json
import pathlib
import shutil # For deleting directory if we choose that route for collection management
import torch # For GPU check

import chromadb # Low-level API, if needed directly
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document # For LangChain Chroma

# Base directory where processed JSON chunk files are stored
CHUNKS_BASE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed_chunks"
# Directory to save the ChromaDB vector store
VECTOR_STORE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "vector_store" / "chroma_db"

# Embedding model name (popular choice, adjust if needed)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# ChromaDB collection name
COLLECTION_NAME = "irs_documents"
BATCH_SIZE = 500  # How many documents to add to Chroma at a time

def get_device():
    """Checks if CUDA is available and returns the appropriate device string."""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for embeddings.")
        return "cuda"
    else:
        print("CUDA not available. Using CPU for embeddings.")
        return "cpu"

def main():
    print(f"Starting vector store build process...")
    CHUNKS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    # VECTOR_STORE_DIR will be created by Chroma if it doesn't exist, 
    # or by the PersistentClient. For deletion, we check its existence.

    device = get_device()
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME} on device: {device}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, 
        model_kwargs={'device': device}
    )

    # --- Collection Management: Delete if exists for a fresh build --- 
    # This ensures that each run starts with a clean slate for the collection.
    # Chroma's persist_directory is where all collections managed by this client would live.
    if VECTOR_STORE_DIR.exists():
        print(f"Found existing vector store directory: {VECTOR_STORE_DIR}")
        # Option 1: Delete the whole directory (simple, effective for local single-collection setup)
        # print(f"Deleting existing vector store directory for a fresh build...")
        # shutil.rmtree(VECTOR_STORE_DIR)
        # VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True) # Recreate after deletion
        
        # Option 2: Use Chroma client to delete the specific collection (more precise)
        try:
            persistent_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
            print(f"Checking for existing collection '{COLLECTION_NAME}'...")
            try:
                collection = persistent_client.get_collection(COLLECTION_NAME)
                if collection:
                    print(f"Deleting existing collection '{COLLECTION_NAME}'...")
                    persistent_client.delete_collection(COLLECTION_NAME)
                    print(f"Collection '{COLLECTION_NAME}' deleted.")
            except Exception as e:
                 # This exception can occur if the collection doesn't exist, which is fine.
                print(f"Collection '{COLLECTION_NAME}' not found or error getting it (which is OK if it's the first run): {e}")
        except Exception as e:
            print(f"Error initializing persistent client or deleting collection: {e}. Proceeding may lead to appending to old data.")
    else:
        print(f"No existing vector store directory found at {VECTOR_STORE_DIR}. A new one will be created.")
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists for Chroma init

    print(f"Initializing ChromaDB vector store at: {VECTOR_STORE_DIR} for collection: {COLLECTION_NAME}")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME, 
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )

    all_docs_to_process = [] 
    all_ids_to_process = []

    # First, collect all documents and IDs from files
    print("Scanning for chunk files and collecting all documents...")
    for year_dir in CHUNKS_BASE_DIR.iterdir():
        if not year_dir.is_dir(): continue
        year = year_dir.name
        print(f"--- Processing year for data collection: {year} ---")
        for json_chunk_file in year_dir.glob("*.json"):
            try:
                with open(json_chunk_file, 'r', encoding='utf-8') as f: chunks_data = json.load(f)
                if not isinstance(chunks_data, list): continue
                for chunk_item in chunks_data:
                    text = chunk_item.get("text")
                    metadata = chunk_item.get("metadata")
                    if not text or not metadata: continue
                    doc = Document(page_content=text, metadata=metadata)
                    all_docs_to_process.append(doc)
                    chunk_id = metadata.get("chunk_id", f"{metadata.get('source_document', 'unknown')}_{metadata.get('year', 'YYYY')}_chunk{len(all_ids_to_process)}")
                    all_ids_to_process.append(chunk_id)
            except Exception as e:
                print(f"Error processing file {json_chunk_file.name}: {e}")
    
    total_docs_to_add = len(all_docs_to_process)
    if total_docs_to_add > 0:
        print(f"Collected {total_docs_to_add} total chunks. Adding to Chroma in batches of {BATCH_SIZE}...")
        
        for i in range(0, total_docs_to_add, BATCH_SIZE):
            batch_docs = all_docs_to_process[i : i + BATCH_SIZE]
            batch_ids = all_ids_to_process[i : i + BATCH_SIZE]
            print(f"Adding batch {i // BATCH_SIZE + 1}/{(total_docs_to_add + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch_docs)} docs)...{batch_ids[:2]}...{batch_ids[-2:]}")
            try:
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                print(f"Successfully added batch of {len(batch_docs)} chunks to the vector store.")
            except Exception as e:
                print(f"Error adding batch to Chroma: {e}")
                # Potentially log batch_ids or details for retry/debug
                # Decide if to continue with next batch or halt
                print("Skipping current batch due to error.") 
        
        print(f"All batches processed. Vector store should be persisted at {VECTOR_STORE_DIR}")
    else:
        print("No chunks found to add to the vector store.")

    print(f"Total chunks intended for processing: {total_docs_to_add}.")
    print("--- Vector store build process complete. ---")

if __name__ == "__main__":
    main() 