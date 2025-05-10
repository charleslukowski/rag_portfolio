import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
# from chromadb.utils import embedding_functions # Not explicitly used
from unstructured.cleaners.core import clean, replace_unicode_quotes
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever # Added for BM25
from langchain_core.documents import Document # Added for creating LC Documents for BM25
import pickle # Added for saving the BM25 retriever

DATA_DIR = Path("../data") if not Path("data").exists() else Path("data")
CHROMA_DIR = "chroma_db"
BM25_RETRIEVER_PATH = Path(CHROMA_DIR) / "bm25_retriever.pkl" # Path to save BM25 object
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Define the headers your document structure uses
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    # Add more if you have "### Header 3", etc.
]

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings())
collection = client.get_or_create_collection("documents")

# Helper: Clean and chunk text
def clean_and_chunk(text: str) -> list[Document]: # Modified to return list of Langchain Documents
    cleaned_text = clean(replace_unicode_quotes(text))

    # First, split by markdown headers
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    md_header_splits = md_splitter.split_text(cleaned_text) # List of LC Documents

    # Then, chunk the content of these documents by character count
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunked_documents = char_splitter.split_documents(md_header_splits) # List of LC Documents
        
    return chunked_documents # Return the Langchain Document objects

# Helper: Extract text from attachments (if present)
def extract_attachment_text(attachment):
    return attachment.get("text", "")

all_lc_documents_for_bm25 = [] # Initialize list to store documents for BM25

# Process all JSON files in data/
for file_idx, file in enumerate(DATA_DIR.glob("*.json")):
    with open(file, encoding="utf-8") as f:
        doc_content_json = json.load(f)
    
    title = doc_content_json.get("title", "")
    author = doc_content_json.get("author", "")
    body = doc_content_json.get("body", "")
    attachments = doc_content_json.get("attachments", [])
    
    # Merge all text
    full_text = f"# {title}\n**Author:** {author}\n\n{body}\n"
    for att in attachments:
        att_text = extract_attachment_text(att)
        if att_text:
            full_text += f"\n## Attachment: {att.get('filename', '')}\n{att_text}\n"
    
    # Clean and chunk into Langchain Document objects
    processed_lc_documents = clean_and_chunk(full_text)
    
    # Process each chunk for ChromaDB and prepare for BM25
    for chunk_idx, doc_from_splitter in enumerate(processed_lc_documents):
        chunk_text = doc_from_splitter.page_content
        
        # Create metadata for this chunk, consistent for both Chroma and BM25 Document
        current_chunk_metadata = doc_from_splitter.metadata.copy() # Start with metadata from splitter (e.g., headers)
        current_chunk_metadata["source_file"] = str(file.name)
        current_chunk_metadata["title"] = title
        current_chunk_metadata["author"] = author
        # current_chunk_metadata["chunk_id"] = f"{file.stem}::p:{chunk_idx}" # For explicit chunk ID if needed

        # Embed and add to ChromaDB
        emb = model.encode(chunk_text)
        chroma_id = f"{file.stem}_{chunk_idx}"
        collection.add(
            embeddings=[emb.tolist()],
            documents=[chunk_text],
            metadatas=[current_chunk_metadata],
            ids=[chroma_id]
        )
        
        # Create a new LangChain Document with the full metadata for BM25
        # (doc_from_splitter already has page_content, we've augmented its metadata in current_chunk_metadata)
        lc_doc_for_bm25 = Document(page_content=chunk_text, metadata=current_chunk_metadata)
        all_lc_documents_for_bm25.append(lc_doc_for_bm25)

    print(f"Processed {file.name}: {len(processed_lc_documents)} chunks added to Chroma and prepared for BM25.")

# After processing all files, create and save BM25 Retriever
if all_lc_documents_for_bm25:
    print(f"\nCreating BM25 Retriever from {len(all_lc_documents_for_bm25)} total chunks...")
    bm25_retriever = BM25Retriever.from_documents(all_lc_documents_for_bm25)
    with open(BM25_RETRIEVER_PATH, "wb") as f_bm25:
        pickle.dump(bm25_retriever, f_bm25)
    print(f"BM25 Retriever created and saved to {BM25_RETRIEVER_PATH}")
else:
    print("\nNo documents were processed, so no BM25 retriever was created.")

print("All documents processed.") 