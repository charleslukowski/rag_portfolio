import os
import json
from pathlib import Path
from document_processor import process_document

RAW_DIR = Path("raw_docs")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Main loop: process each file in raw_docs/
for file in RAW_DIR.iterdir():
    if not file.is_file():
        continue
    
    # Use the process_document function from document_processor.py
    # This will handle supported file types and cleaning as defined in document_processor.py
    cleaned_text = process_document(str(file), verbose=False) 
    
    if not cleaned_text: # process_document returns None on failure or unsupported type
        # The process_document function already prints messages for unsupported/error cases
        continue
        
    doc_json = {
        "title": file.stem,
        "author": "",  # Could be extended with more metadata
        "body": cleaned_text, # Use the cleaned text
        "attachments": []
    }
    out_path = DATA_DIR / f"{file.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc_json, f, indent=2, ensure_ascii=False)
    print(f"Extracted and cleaned {file.name} -> {out_path.name}")

print("All processable raw documents extracted to JSON.") 