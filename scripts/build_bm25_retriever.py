import os
import json
import pickle
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Define paths
PROCESSED_CHUNKS_ROOT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_chunks"
BM25_RETRIEVER_DIR = Path(__file__).resolve().parent.parent / "data" / "bm25_retriever"
BM25_RETRIEVER_PATH = BM25_RETRIEVER_DIR / "irs_bm25_retriever.pkl"

MIN_CHUNK_SIZE_FOR_BM25 = 10 # Consider if a different minimum size is needed for BM25

def load_all_chunks():
    """Loads all processed chunks from the data/processed_chunks directory."""
    all_docs = []
    if not PROCESSED_CHUNKS_ROOT_DIR.exists():
        print(f"Error: Processed chunks directory not found at {PROCESSED_CHUNKS_ROOT_DIR}")
        return []

    for year_dir in PROCESSED_CHUNKS_ROOT_DIR.iterdir():
        if year_dir.is_dir():
            year = year_dir.name
            for chunk_file_path in year_dir.glob("*.json"):
                try:
                    with open(chunk_file_path, 'r', encoding='utf-8') as f:
                        # chunk_list_from_file is a list of dictionaries, where each dict is a chunk
                        chunk_list_from_file = json.load(f)
                        
                        for chunk_info in chunk_list_from_file: # Iterate directly over the list of chunks
                            text = chunk_info.get("text")
                            # metadata from the chunk as saved by preprocess_text.py
                            processed_metadata = chunk_info.get("metadata", {})

                            # Construct metadata for the Langchain Document
                            doc_metadata = {}
                            doc_metadata['year'] = year # 'year' is from year_dir.name (e.g., "2024")
                            
                            # 'source_document' should be in processed_metadata. Fallback to filename stem.
                            doc_metadata['source_document'] = processed_metadata.get('source_document', chunk_file_path.stem)
                            
                            # Carry over other useful metadata if present
                            if 'chunk_id' in processed_metadata:
                                doc_metadata['chunk_id'] = processed_metadata['chunk_id']
                            if 'heading_text' in processed_metadata: # As per preprocess_text.py summary
                                doc_metadata['heading_text'] = processed_metadata['heading_text']

                            if text and len(text) >= MIN_CHUNK_SIZE_FOR_BM25:
                                all_docs.append(Document(page_content=text, metadata=doc_metadata))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {chunk_file_path}")
                except Exception as e:
                    print(f"Error processing file {chunk_file_path}: {e}")
    
    print(f"Loaded {len(all_docs)} documents for BM25 retriever.")
    return all_docs

def build_and_save_bm25_retriever(documents):
    """Builds the BM25 retriever and saves it to disk."""
    if not documents:
        print("No documents found to build BM25 retriever. Exiting.")
        return

    print("Initializing BM25Retriever...")
    # BM25Retriever.from_documents expects a List[Document]
    # It internally handles tokenization and indexing.
    bm25_retriever = BM25Retriever.from_documents(documents)
    print("BM25Retriever initialized.")

    # Ensure the save directory exists
    BM25_RETRIEVER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving BM25Retriever to {BM25_RETRIEVER_PATH}...")
    try:
        with open(BM25_RETRIEVER_PATH, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        print("BM25Retriever saved successfully.")
    except Exception as e:
        print(f"Error saving BM25Retriever: {e}")

def main():
    print("Starting BM25 retriever build process...")
    documents = load_all_chunks()
    if documents:
        build_and_save_bm25_retriever(documents)
    else:
        print("No documents were loaded. BM25 retriever build aborted.")

if __name__ == "__main__":
    main() 