import pathlib
import re
# Remove direct imports of docx, openpyxl, pptx, pdfplumber as unstructured will handle them
# import docx 
# import openpyxl
# import pptx 
# import pdfplumber
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter # Added for chunking

# The individual _extract_text_from_... functions will be removed.

def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing whitespace, removing control characters,
    and preparing it for further processing.
    """
    # 1. Normalize whitespace: replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # 2. Remove common control characters (keeping newline, carriage return, tab for now)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            cleaned_lines.append(stripped_line)
    
    return "\n".join(cleaned_lines)

def process_document(file_path_str: str, verbose: bool = True) -> str | None:
    """
    Processes a single document file for text extraction and cleaning using 'unstructured'.
    If verbose is False, it will only print concise error or skip messages.
    """
    file_path = pathlib.Path(file_path_str)
    if not file_path.is_file():
        message = f"Error processing {file_path.name}: File not found at {file_path}"
        if verbose:
            print(message)
        else:
            print(message) # Critical error, always print
        return None

    file_extension = file_path.suffix.lower()
    supported_extensions = ['.docx', '.xlsx', '.pptx', '.pdf', '.doc', '.xls']

    if file_extension not in supported_extensions:
        message = f"Skipping unsupported file: {file_path.name} (Type: {file_extension})"
        if verbose:
            print(f"Unsupported file type: {file_extension} for file {file_path.name}. Skipping.")
        else:
            print(message)
        return None

    if verbose:
        print(f"Processing document with unstructured: {file_path.name} (Type: {file_extension})")

    try:
        elements = partition(filename=str(file_path))
        raw_text = "\n\n".join([el.text for el in elements if hasattr(el, 'text') and el.text])
    except Exception as e:
        # Extract a more concise error message if possible, e.g. the part about soffice
        error_summary = str(e).split('\n')[0] # Get the first line of a multi-line error
        message = f"Error processing {file_path.name}: {error_summary}"
        if verbose:
            print(f"Error using unstructured to extract text from {file_path.name}: {e}")
        else:
            print(message)
        return None

    if not raw_text.strip():
        message = f"Error processing {file_path.name}: No text could be extracted by unstructured."
        if verbose:
            print(f"No text could be extracted by unstructured from {file_path.name}")
        else:
            print(message)
        return None

    if verbose:
        print(f"Successfully extracted raw text with unstructured from {file_path.name} (Length: {len(raw_text)} chars)")
    
    cleaned_text = clean_text(raw_text)
    if verbose:
        print(f"Cleaned text length for {file_path.name}: {len(cleaned_text)} chars")
    
    return cleaned_text

def chunk_cleaned_text(text: str) -> list[str]:
    """
    Splits cleaned text into chunks using RecursiveCharacterTextSplitter.
    Uses character count for chunk_size and chunk_overlap as per refactor.md example.
    """
    if not text:
        print("Cannot chunk empty text.")
        return []

    # Separators are ordered from most semantically significant to least.
    # chunk_size and chunk_overlap are in characters by default.
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n\n", ". ", " ", ""],
        chunk_size=600,  # Characters
        chunk_overlap=120, # Characters
        keep_separator=True, # Keep separators at the start of the chunk for context
        strip_whitespace=True
    )
    chunks = splitter.split_text(text)
    print(f"Successfully split text into {len(chunks)} chunks.")
    return chunks

if __name__ == '__main__':
    # Example usage (assuming you have files in a 'raw_docs' directory sibling to 'scripts')
    # Create a 'raw_docs' directory in your project root and place some sample files there.
    # For example:
    # raw_docs/
    #   ├── my_document.docx
    #   ├── my_spreadsheet.xlsx
    #   ├── my_presentation.pptx
    #   └── my_report.pdf

    sample_docs_dir = pathlib.Path(__file__).parent.parent / "raw_docs"
    
    if not sample_docs_dir.exists():
        print(f"Creating sample directory: {sample_docs_dir} for example usage.")
        sample_docs_dir.mkdir(exist_ok=True)
        print(f"Please add supported files (.docx, .xlsx, .pptx, .pdf) to {sample_docs_dir} to test.")
    else:
        print(f"Looking for sample documents in: {sample_docs_dir}")
        for doc_file in sample_docs_dir.iterdir():
            if doc_file.is_file():
                print(f"\n--- Attempting to process and chunk: {doc_file.name} ---")
                cleaned_content = process_document(str(doc_file))
                if cleaned_content:
                    chunks = chunk_cleaned_text(cleaned_content)
                    if chunks:
                        print(f"First chunk from {doc_file.name} (first 100 chars):\n'{chunks[0][:100]}...'")
                        # You might want to do something with all chunks here, like print details or save them
                else:
                    print(f"Could not process {doc_file.name}, skipping chunking.") 