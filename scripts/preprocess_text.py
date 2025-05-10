import os
import json
import pathlib
import re

# Base directory where processed .txt files are stored
PROCESSED_TEXT_BASE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed_text"
# Base directory to save processed chunk files
CHUNKS_BASE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed_chunks"

# Regex patterns for potential headings (can be expanded)
HEADING_PATTERNS = [
    re.compile(r"^CHAPTER\s+\d+[\s.:\-]+\s*(.*)", re.IGNORECASE),
    re.compile(r"^PART\s+[IVXLCDM\d]+[\s.:\-]+\s*(.*)", re.IGNORECASE), # Roman numerals or digits for Part
    re.compile(r"^SECTION\s+\d+[\s.:\-]+\s*(.*)", re.IGNORECASE),
    re.compile(r"^APPENDIX\s+[A-Z\d]+[\s.:\-]+\s*(.*)", re.IGNORECASE),
    re.compile(r"^Table of Contents", re.IGNORECASE),
    re.compile(r"^Introduction", re.IGNORECASE),
    re.compile(r"^What's New", re.IGNORECASE),
    re.compile(r"^Reminders", re.IGNORECASE),
    re.compile(r"^Index", re.IGNORECASE),
    # A more generic one for all-caps lines (potential headings)
    # This needs to be handled carefully to avoid splitting too much.
    # Consider lines with 2-5 words, all uppercase.
    # re.compile(r"^([A-Z][A-Z\s]{5,80}[A-Z])(?:\n|\Z)") # Example: A short, all-caps line
]

MIN_CHUNK_SIZE = 200 # Minimum characters for a chunk to be considered somewhat substantial

def identify_headings_and_split(text_lines):
    """Identifies lines that look like headings and uses them as split points.
    Returns a list of tuples: (list_of_lines_for_chunk, heading_text_or_None)
    """
    processed_chunks = [] # Stores (list_of_lines, heading_text)
    current_chunk_lines = []
    current_chunk_has_explicit_heading = None

    for line in text_lines:
        is_heading_line = False
        identified_heading_text = None
        for pattern in HEADING_PATTERNS:
            match = pattern.match(line)
            if match:
                is_heading_line = True
                identified_heading_text = line.strip() # The heading itself
                # print(f"Found heading: {identified_heading_text} by pattern {pattern.pattern}")
                break
        
        if is_heading_line:
            if current_chunk_lines: # If we have an accumulated chunk, save it
                processed_chunks.append((current_chunk_lines, current_chunk_has_explicit_heading))
            current_chunk_lines = [line] # New chunk starts with this heading line
            current_chunk_has_explicit_heading = identified_heading_text
        else:
            current_chunk_lines.append(line)
            
    if current_chunk_lines: # Add the last accumulated chunk
        processed_chunks.append((current_chunk_lines, current_chunk_has_explicit_heading))
    
    # Convert line lists to text and filter small chunks
    final_output_chunks = []
    for lines_list, heading_txt in processed_chunks:
        chunk_text_content = "\n".join(lines_list).strip()
        # Keep chunk if substantial OR if it explicitly started with one of our known heading patterns
        if len(chunk_text_content) > MIN_CHUNK_SIZE or heading_txt:
            final_output_chunks.append((chunk_text_content, heading_txt))
            
    return final_output_chunks

def split_by_paragraphs(text):
    """Splits text by double newlines (paragraphs)."""
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > MIN_CHUNK_SIZE / 2] # Lower min for paragraphs

def chunk_document_text(doc_text, source_doc_name, year):
    """Chunks document text using a chosen strategy."""
    lines = doc_text.splitlines()
    
    # Strategy 1: Try splitting by identified headings
    # This now returns list of (chunk_text_content, heading_text_or_None)
    chunks_with_headings = identify_headings_and_split(lines)
    
    final_chunks_data = []

    if chunks_with_headings and len(chunks_with_headings) > 1: 
        # print(f"Using heading-based splitting for {source_doc_name}")
        for i, (chunk_text, heading_text) in enumerate(chunks_with_headings):
            metadata = {
                "source_document": source_doc_name,
                "year": year,
                "chunk_id": f"{source_doc_name}_{year}_h{i}",
                # "page_number(s)": "TBD" 
            }
            if heading_text:
                metadata["heading"] = heading_text
            final_chunks_data.append({"text": chunk_text, "metadata": metadata})
    else:
        # print(f"Using paragraph-based splitting for {source_doc_name}")
        paragraph_chunks = split_by_paragraphs(doc_text)
        for i, chunk_text in enumerate(paragraph_chunks):
            final_chunks_data.append({
                "text": chunk_text,
                "metadata": {
                    "source_document": source_doc_name,
                    "year": year,
                    "chunk_id": f"{source_doc_name}_{year}_p{i}",
                }
            })
            
    if not final_chunks_data and doc_text.strip():
        # print(f"No chunks generated by primary methods for {source_doc_name}, saving as single chunk.")
        final_chunks_data.append({
            "text": doc_text.strip(),
            "metadata": {
                "source_document": source_doc_name,
                "year": year,
                "chunk_id": f"{source_doc_name}_{year}_s0",
            }
        })

    return final_chunks_data

def main():
    print("Starting text preprocessing and chunking...")
    CHUNKS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    for year_dir in PROCESSED_TEXT_BASE_DIR.iterdir():
        if not year_dir.is_dir():
            continue
        
        year = year_dir.name
        output_year_chunks_dir = CHUNKS_BASE_DIR / year
        output_year_chunks_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- Processing year: {year} ---")
        
        processed_files_count = 0
        for txt_file_path in year_dir.glob("*.txt"):
            print(f"Processing file: {txt_file_path.name}")
            try:
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
            except IOError as e:
                print(f"Error reading {txt_file_path}: {e}. Skipping.")
                continue

            source_doc_name = txt_file_path.stem
            chunked_data = chunk_document_text(raw_text, source_doc_name, year)
            
            if chunked_data:
                output_json_filename = source_doc_name + ".json"
                output_json_path = output_year_chunks_dir / output_json_filename
                try:
                    with open(output_json_path, 'w', encoding='utf-8') as json_file:
                        json.dump(chunked_data, json_file, indent=2, ensure_ascii=False)
                    # print(f"Saved {len(chunked_data)} chunks to: {output_json_path}")
                    processed_files_count += 1
                except IOError as e:
                    print(f"Error writing JSON file {output_json_path}: {e}")
            else:
                print(f"No chunks generated for {txt_file_path.name}.")
        
        print(f"Processed and chunked {processed_files_count} text file(s) for the year {year}.")

    print("--- Text preprocessing and chunking complete. ---")

if __name__ == "__main__":
    main() 