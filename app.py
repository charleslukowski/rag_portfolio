import streamlit as st
# import chromadb # We'll use Langchain's Chroma wrapper
# from chromadb.config import Settings # Not needed with Langchain's Chroma
# from sentence_transformers import SentenceTransformer # Langchain will wrap this
from pathlib import Path
import pickle # Added for loading the BM25 retriever
import os # Added for environment variables
from dotenv import load_dotenv # Added for .env file loading

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever # Added for BM25
from langchain.retrievers import EnsembleRetriever # Added for combining
from langchain_core.documents import Document # Added for dummy BM25 docs
from sentence_transformers import CrossEncoder # Added for reranking
from langchain_openai import ChatOpenAI # Added for LLM
from langchain_core.prompts import ChatPromptTemplate # Added for prompt engineering
from langchain_core.runnables import RunnablePassthrough # Added for chain construction
from langchain_core.output_parsers import StrOutputParser # Added for parsing LLM output

import pathlib
import torch # For GPU check
from typing import List, Optional, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(page_title="IRS Document RAG Advisor", layout="wide")

# --- Load Environment Variables ---
load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Not strictly needed here if langchain handles it

# --- Configuration for IRS RAG Project ---
# Directory where the ChromaDB for IRS documents is stored (consistent with build_vector_store.py)
IRS_VECTOR_STORE_DIR = pathlib.Path(__file__).resolve().parent / "data" / "vector_store" / "chroma_db"
# Embedding model name (consistent with build_vector_store.py)
IRS_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# ChromaDB collection name for IRS documents (consistent with build_vector_store.py)
IRS_COLLECTION_NAME = "irs_documents"
# Path for the pickled BM25 retriever
BM25_IRS_RETRIEVER_PATH = pathlib.Path(__file__).resolve().parent / "data" / "bm25_retriever" / "irs_bm25_retriever.pkl"

N_RESULTS_TO_RETRIEVE = 3  # Number of documents to retrieve initially (reduced from 5)
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2' # Added for reranking
LLM_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano-2025-04-14")

# Global variable for the main retriever
main_retriever_instance: Optional[EnsembleRetriever] = None # Will be the EnsembleRetriever
# Global variables for other models that will be loaded
# We define them here so they can be updated by load_irs_retriever_and_models
# and used by the RAG chain and UI components.
llm: Optional[ChatOpenAI] = None
cross_encoder: Optional[CrossEncoder] = None
# We will also store the individual retrievers in case they are needed.
irs_vector_store_loaded: Optional[Chroma] = None
bm25_retriever_loaded: Optional[BM25Retriever] = None

def get_device_for_embeddings():
    """Checks if CUDA is available and returns the appropriate device string."""
    if torch.cuda.is_available():
        print("CUDA available. Using GPU for embeddings.")
        return "cuda"
    else:
        print("CUDA not available. Using CPU for embeddings.")
        return "cpu"

@st.cache_resource
def load_irs_retriever_and_models():
    """Loads the Chroma vector store, BM25 retriever, LLM, and CrossEncoder models.
    Returns a tuple: (ensemble_retriever, loaded_llm, loaded_cross_encoder, success_flag)
    """
    # Temp local variables for loading
    local_main_retriever: Optional[EnsembleRetriever] = None
    local_llm: Optional[ChatOpenAI] = None
    local_cross_encoder: Optional[CrossEncoder] = None
    local_irs_vector_store: Optional[Chroma] = None
    local_bm25_retriever: Optional[BM25Retriever] = None
    
    print("Attempting to load IRS resources (inside @st.cache_resource function)...")

    # 1. Load Chroma Vector Store
    if not IRS_VECTOR_STORE_DIR.exists():
        st.error(f"IRS Vector store directory not found at {IRS_VECTOR_STORE_DIR}.")
        return None, None, None, False
    try:
        device = get_device_for_embeddings()
        embeddings = HuggingFaceEmbeddings(model_name=IRS_EMBEDDING_MODEL_NAME, model_kwargs={'device': device})
        local_irs_vector_store = Chroma(
            collection_name=IRS_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(IRS_VECTOR_STORE_DIR)
        )
        print(f"IRS Chroma vector store loaded. Docs: {local_irs_vector_store._collection.count()}")
    except Exception as e:
        st.error(f"Error loading IRS vector store: {e}")
        return None, None, None, False

    # 2. Load BM25 Retriever
    if not BM25_IRS_RETRIEVER_PATH.exists():
        st.error(f"BM25 retriever file not found at {BM25_IRS_RETRIEVER_PATH}.")
        return None, None, None, False
    try:
        with open(BM25_IRS_RETRIEVER_PATH, 'rb') as f:
            local_bm25_retriever = pickle.load(f)
        print("BM25 retriever loaded successfully.")
    except Exception as e:
        st.error(f"Error loading BM25 retriever: {e}")
        return None, None, None, False

    # 3. Initialize Ensemble Retriever
    if local_irs_vector_store and local_bm25_retriever:
        chroma_as_retriever = local_irs_vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": N_RESULTS_TO_RETRIEVE}
        )
        local_bm25_retriever.k = N_RESULTS_TO_RETRIEVE
        local_main_retriever = EnsembleRetriever(
            retrievers=[chroma_as_retriever, local_bm25_retriever],
            weights=[0.5, 0.5]
        )
        print("EnsembleRetriever initialized.")
    else:
        st.error("Failed to initialize EnsembleRetriever (missing vector store or BM25).")
        return None, None, None, False

    # 4. Load CrossEncoder Model
    try:
        local_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        print("CrossEncoder model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading CrossEncoder model: {e}")
        return local_main_retriever, None, None, False # Return what we have so far

    # 5. Load LLM Model
    try:
        local_llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0)
        print("LLM model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading LLM (OpenAI): {e}.")
        return local_main_retriever, None, local_cross_encoder, False # Return what we have
    
    print("All IRS-focused resources loaded successfully by @st.cache_resource function.")
    # Return the loaded models explicitly
    return local_main_retriever, local_llm, local_cross_encoder, True

# Load resources once when the app starts and assign to globals
main_retriever_instance, llm, cross_encoder, resources_loaded_successfully = load_irs_retriever_and_models()

# --- Prompt Template for Grounded QA ---
prompt_template_str = """\
Answer the user's question based ONLY on the provided excerpts from IRS documents.
Cite the source document (e.g., Pub 17, Form 1040 Instructions) and the year for each piece of information you use.
If the answer cannot be found in the excerpts, state that clearly.

Excerpts:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template_str)

# --- RAG Chain ---
# Define the RAG chain. It will use the globally loaded `llm`.
# Ensure this is defined *after* `llm` is potentially loaded.
# If `llm` is None at this point (e.g. load failed), it will cause issues later.
# A better way is to build the chain inside the button click if llm is loaded.
# For now, we rely on Streamlit's top-to-bottom execution and `llm` being populated.

rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()} 
    | prompt
    | (lambda x: llm if llm else "LLM NOT LOADED") # Defensive check for llm
    | StrOutputParser()
)

# --- Helper function to format documents for context ---
def format_docs_for_context(docs: List[Document]):
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source_document', 'Unknown Source')
        year = doc.metadata.get('year', 'Unknown Year')
        # Use 'heading_text' as populated by build_bm25_retriever.py and potentially build_vector_store.py
        heading = doc.metadata.get('heading_text', doc.metadata.get('heading', None)) # Fallback to 'heading' if 'heading_text' not present
        content = doc.page_content
        header = f"(Source: {source}, Year: {year}, Excerpt {i+1})"
        if heading:
            header += f" (From section: {heading[:100]})" # Truncate long headings
        formatted_docs.append(f"{header}\\n{content}")
    return "\\n\\n---\\n\\n".join(formatted_docs)

# --- Streamlit App UI ---
st.title("IRS Document RAG Advisor")
st.write("Ask questions about IRS forms and publications. The system uses vector search, reranking, and an LLM to generate answers based on document excerpts.")

# --- Sidebar for Controls ---
st.sidebar.header("Controls")

# Year selection
available_years = sorted([d.name for d in (pathlib.Path(__file__).resolve().parent / "data" / "processed_chunks").iterdir() if d.is_dir()], reverse=True)
if not available_years:
    available_years = ["2024"] # Default if no processed years found

options = ["None (all years)"] + available_years
default_index = 0 # Default to "None (all years)"
if "2024" in options:
    default_index = options.index("2024")
elif available_years: # If 2024 not present, but other years are, default to the latest one
    # options[0] is "None (all years)", options[1] is the latest year
    if len(options) > 1:
        default_index = 1


selected_year = st.sidebar.selectbox(
    "Select Tax Year:", 
    options=options,
    index=default_index,
    help="Select a specific tax year for your query, or search all available years."
)

user_query = st.sidebar.text_input("Enter your query:", placeholder="e.g., mileage rate for business car")
submit_button = st.sidebar.button("Get IRS Guidance")

# --- Main Page for Results ---
if submit_button:
    if not resources_loaded_successfully or not main_retriever_instance or not cross_encoder or not llm:
        st.error("Core resources (Retriever, CrossEncoder, or LLM) are not available. Please check application startup logs.")
    elif user_query:
        st.subheader("LLM Answer:")
        with st.spinner("Searching IRS documents, reranking, and generating answer..."):
            
            # 1. Initial retrieval from EnsembleRetriever
            # k for sub-retrievers was set during EnsembleRetriever initialization.
            query_to_log = user_query # To avoid issues with f-string if query has braces
            print(f"Retrieving from EnsembleRetriever with query: '{query_to_log}'")
            
            try:
                initial_results = main_retriever_instance.get_relevant_documents(user_query)
            except Exception as e:
                st.error(f"Error during document retrieval: {e}")
                initial_results = []
            
            # 2. Post-retrieval year filtering if a year is selected
            query_year = None
            if selected_year != "None (all years)":
                query_year = selected_year
                print(f"Applying post-retrieval filter for year: {query_year}")
                
                if not initial_results: # already empty, no need to filter
                    print("Initial results empty, skipping year filter.")
                else:
                    filtered_results_after_year = []
                    for doc in initial_results:
                        if doc.metadata.get("year") == query_year:
                            filtered_results_after_year.append(doc)
                    initial_results = filtered_results_after_year # Update initial_results
                    print(f"Number of results after year filter: {len(initial_results)}")

            if not initial_results:
                st.info("No relevant documents found in the IRS knowledge base for your query (after potential year filtering).")
            else:
                # 3. Reranking
                sentence_pairs = [[user_query, doc.page_content] for doc in initial_results]
                try:
                    # Use the globally loaded cross_encoder
                    scores = cross_encoder.predict(sentence_pairs)
                    reranked_results_with_scores = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)
                    reranked_docs = [doc for doc, score in reranked_results_with_scores]
                    print(f"Reranked {len(reranked_docs)} documents.")
                except Exception as e:
                    st.error(f"Error during reranking: {e}")
                    reranked_docs = initial_results # Fallback to non-reranked if error

            if reranked_docs:
                # 4. Format context for LLM
                context_str = format_docs_for_context(reranked_docs)
                st.markdown("### Retrieved Excerpts (after reranking):")
                with st.expander("View Excerpts Used for Answer", expanded=False):
                    st.text_area("Context:", context_str, height=300)
                
                # 5. Generate answer with LLM
                try:
                    # rag_chain uses the globally loaded llm
                    answer = rag_chain.invoke({"context": context_str, "question": user_query})
                    st.markdown("#### LLM Answer:") # Changed from st.subheader to st.markdown for consistency
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error generating answer with LLM: {e}")
    else:
        st.info("Please enter a query.")

# Placeholder for the old example queries (can be removed or adapted if __name__ == "__main__" is added for app.py testing)
# if __name__ == "__main__":
#     print("Testing RAG retriever functionality (not running Streamlit app)...")
#     # Test loading
#     if load_irs_retriever_and_models(): # This will now set global vars
#         print("Models loaded for testing.")
#         if main_retriever_instance and llm and cross_encoder:
#             query1 = "what is the mileage rate for business use of a car"
#             
#             # Test Ensemble Retrieval (no year filter at this stage for ensemble)
#             ensemble_docs = main_retriever_instance.get_relevant_documents(query1)
#             print(f"\n--- Ensemble Test Query: '{query1}' --- ({len(ensemble_docs)} results)")
#             for doc in ensemble_docs:
#                 print(f"  Source: {doc.metadata.get('source_document')}, Year: {doc.metadata.get('year')}, Content: {doc.page_content[:70]}...")
#             
#             # Test Post-Retrieval Year Filtering (Simulating UI logic)
#             year_to_filter = "2024"
#             year_filtered_docs = [doc for doc in ensemble_docs if doc.metadata.get("year") == year_to_filter]
#             print(f"\n--- Filtered for Year {year_to_filter} ({len(year_filtered_docs)} results) --- ")
#             for doc in year_filtered_docs:
#                 print(f"  Source: {doc.metadata.get('source_document')}, Year: {doc.metadata.get('year')}, Content: {doc.page_content[:70]}...")
# 
#             # Test Reranking (using a subset of year_filtered_docs or ensemble_docs)
#             if year_filtered_docs:
#                 rerank_sentence_pairs = [[query1, doc.page_content] for doc in year_filtered_docs[:5]] # Rerank top 5
#                 rerank_scores = cross_encoder.predict(rerank_sentence_pairs)
#                 reranked_with_scores = sorted(zip(year_filtered_docs[:5], rerank_scores), key=lambda x: x[1], reverse=True)
#                 print(f"\n--- Reranked for Year {year_to_filter} ({len(reranked_with_scores)} results) ---")
#                 for doc, score in reranked_with_scores:
#                     print(f"  Score: {score:.4f}, Source: {doc.metadata.get('source_document')}, Year: {doc.metadata.get('year')}, Content: {doc.page_content[:70]}...")
# 
#             # Test RAG Chain
#             if reranked_with_scores: # Use reranked for context
#                 context_for_llm = format_docs_for_context([doc for doc, score in reranked_with_scores])
#                 print(f"\n--- LLM Context (Year {year_to_filter}, Reranked) ---")
#                 # print(context_for_llm) # Can be very verbose
#                 print("Invoking RAG chain...")
#                 llm_answer = rag_chain.invoke({"context": context_for_llm, "question": query1})
#                 print(f"\n--- LLM Answer (Year {year_to_filter}, Reranked) ---")
#                 print(llm_answer)
# 
#         else:
#             print("Main retriever, LLM, or CrossEncoder instance not available for testing.")
#     else:
#         print("Models could not be loaded for testing.")

