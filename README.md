# IRS Document RAG Advisor

This project implements a RAG (Retrieval Augmented Generation) system to answer questions about IRS (Internal Revenue Service) forms and publications. It uses a Streamlit interface for user interaction, LangChain for orchestrating the RAG pipeline, ChromaDB for vector storage, BM25 for keyword-based retrieval, and an OpenAI LLM for generating answers.

## Features

*   Retrieves relevant excerpts from a collection of IRS PDF documents.
*   Uses an ensemble retriever (Chroma vector search + BM25 keyword search).
*   Reranks retrieved documents using a CrossEncoder model.
*   Generates answers based on the retrieved context using an OpenAI LLM.
*   Allows filtering by tax year.
*   Streamlit UI for easy interaction.

## Project Structure

```
rag_portfolio/
├── .venv/                  # Virtual environment
├── data/
│   ├── bm25_retriever/     # Stores the pickled BM25 retriever
│   ├── processed_chunks/   # Stores JSON files of processed text chunks
│   ├── processed_text/     # Stores raw extracted text from PDFs
│   ├── raw_pdfs/           # Stores downloaded IRS PDF documents
│   └── vector_store/       # Stores the ChromaDB vector store
├── scripts/                # Python scripts for data processing and RAG components
│   ├── download_irs_docs.py
│   ├── extract_pdf_text.py
│   ├── preprocess_text.py
│   ├── build_vector_store.py
│   └── build_bm25_retriever.py
├── .env.example            # Example environment file
├── .gitignore
├── app.py                  # Main Streamlit application
└── requirements.txt        # Python dependencies
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd rag_portfolio
    ```

2.  **Create and activate a Python virtual environment:**
    *   On Windows:
        ```bash
        python -m venv .venv
        .venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    *   Create a `.env` file in the project root directory by copying `.env.example`:
        ```bash
        # On Windows
        copy .env.example .env 
        # On macOS/Linux
        # cp .env.example .env
        ```
    *   Edit the `.env` file and add your OpenAI API key and desired model:
        ```
        OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
        OPENAI_MODEL_NAME="gpt-4.1-nano-2025-04-14" # Or your preferred model
        ```

## Data Preparation

Run the following scripts in order from the `rag_portfolio` root directory to download IRS documents, process them, and build the retrieval components.

**Note:** The `download_irs_docs.py` script currently downloads a limited set of documents and years for testing. You can modify the `DOC_CODES` and `YEARS` variables in the script to download a broader range of documents.

1.  **Download IRS PDF documents:**
    ```bash
    python scripts/download_irs_docs.py
    ```
    This will download PDFs into `data/raw_pdfs/YEAR/` and create a `manifest.json` in each year's directory.

2.  **Extract text from PDFs:**
    ```bash
    python scripts/extract_pdf_text.py
    ```
    This reads PDFs listed in the manifests and saves extracted text to `data/processed_text/YEAR/`.

3.  **Preprocess and chunk text:**
    ```bash
    python scripts/preprocess_text.py
    ```
    This reads the extracted text, chunks it, and saves the chunks as JSON files in `data/processed_chunks/YEAR/`.

4.  **Build the Chroma vector store:**
    ```bash
    python scripts/build_vector_store.py
    ```
    This creates embeddings for the chunks and stores them in `data/vector_store/chroma_db/`.

5.  **Build the BM25 retriever:**
    ```bash
    python scripts/build_bm25_retriever.py
    ```
    This creates and pickles a BM25 retriever model to `data/bm25_retriever/irs_bm25_retriever.pkl`.

## Running the Application

Once the setup and data preparation steps are complete, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your web browser.

## Customization

*   **IRS Documents:** Modify `DOC_CODES` and `YEARS` in `scripts/download_irs_docs.py` to target different IRS forms/publications and years.
*   **Embedding Model:** The embedding model for ChromaDB is set in `scripts/build_vector_store.py` and `app.py` (currently `all-MiniLM-L6-v2`).
*   **LLM Model:** The OpenAI LLM model is specified via the `OPENAI_MODEL_NAME` environment variable in your `.env` file.
*   **Chunking Strategy:** The chunking logic is in `scripts/preprocess_text.py`.
*   **Retrieval Parameters:** Parameters like the number of documents to retrieve (`N_RESULTS_TO_RETRIEVE`) can be adjusted in `app.py`. 