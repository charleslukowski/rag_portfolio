import os
import json
import pathlib
import fitz  # PyMuPDF

# Base directory where downloaded PDFs and their manifests are stored
RAW_PDF_BASE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw_pdfs"
# Base directory to save extracted plain text files
PROCESSED_TEXT_BASE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed_text"

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a given PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return None

def main():
    print(f"Starting PDF text extraction...")
    PROCESSED_TEXT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Iterate through year directories in the raw_pdfs directory
    for year_dir in RAW_PDF_BASE_DIR.iterdir():
        if not year_dir.is_dir():
            continue
        
        year = year_dir.name
        manifest_path = year_dir / "manifest.json"

        if not manifest_path.exists():
            print(f"No manifest.json found in {year_dir}. Skipping.")
            continue

        print(f"--- Processing year: {year} ---")
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding {manifest_path}: {e}. Skipping year {year}.")
            continue
        except IOError as e:
            print(f"Error reading {manifest_path}: {e}. Skipping year {year}.")
            continue

        output_year_dir = PROCESSED_TEXT_BASE_DIR / year
        output_year_dir.mkdir(parents=True, exist_ok=True)
        
        processed_files_count = 0
        for item in manifest_data:
            # The manifest from download_irs_docs.py (latest version) has "local", "url", "year".
            # It does not have a separate "filename" key. We derive filename from the "local" path.
            pdf_local_path_str = item.get("local")
            if not pdf_local_path_str:
                print(f"Skipping item with no local_path in manifest: {item}")
                continue

            pdf_path = pathlib.Path(pdf_local_path_str)
            pdf_actual_filename = pdf_path.name # Get the filename for messages

            # Removed the check for item.get("filename") as it's not in the current manifest structure
            # and we can derive the filename from the local path.

            if not pdf_path.exists():
                print(f"PDF file {pdf_path} ({pdf_actual_filename}) listed in manifest not found. Skipping.")
                continue

            print(f"Extracting text from: {pdf_actual_filename}")
            raw_text = extract_text_from_pdf(pdf_path)

            if raw_text:
                output_txt_filename = pdf_path.stem + ".txt"
                output_txt_path = output_year_dir / output_txt_filename
                try:
                    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(raw_text)
                    # print(f"Saved extracted text to: {output_txt_path}")
                    processed_files_count += 1
                except IOError as e:
                    print(f"Error writing text file {output_txt_path}: {e}")
            else:
                print(f"Failed to extract text from {pdf_path.name}.")
        
        print(f"Processed {processed_files_count} PDF(s) for the year {year}.")

    print("--- PDF text extraction complete. ---")

if __name__ == "__main__":
    main() 