# ingestion.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import pypdf
import argparse
import os

# --- Helper function for chunking text ---
def chunk_text(text, chunk_size=300, overlap=50):
    """Splits text into chunks of a specified size with overlap."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        start_pos = current_pos
        end_pos = current_pos + chunk_size
        chunk_words = words[start_pos:end_pos]
        chunks.append(" ".join(chunk_words))
        
        current_pos += chunk_size - overlap
        if current_pos >= len(words):
            break
            
    return chunks

# --- Main Ingestion Logic ---
def process_pdf(pdf_path: str):
    """
    Reads a PDF, chunks its text, creates embeddings, and saves them to a FAISS index.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return

    print(f"Processing PDF: {pdf_path}...")
    pdf_filename = os.path.basename(pdf_path)
    
    # 1. Extract Text from PDF
    try:
        reader = pypdf.PdfReader(pdf_path)
        all_chunks = []
        all_metadata = []

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                page_chunks = chunk_text(text)
                for chunk in page_chunks:
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "pdf_id": pdf_filename,
                        "page": page_num,
                        "content": chunk
                    })
        
        if not all_chunks:
            print("Warning: No text could be extracted from the PDF.")
            return
            
    except Exception as e:
        print(f"Error reading or processing PDF: {e}")
        return

    # 2. Initialize Embedding Model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 

    # 3. Generate Embeddings
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')

    # 4. Build and Save FAISS Index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index and metadata
    faiss.write_index(index, "pdf_index.faiss")
    with open("metadata.json", "w") as f:
        json.dump(all_metadata, f)

    print("\n✅ Ingestion complete!")
    print("'pdf_index.faiss' and 'metadata.json' have been updated with your PDF content.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF to create a searchable index.")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the PDF file.")
    args = parser.parse_args()
    
    process_pdf(args.pdf_path)