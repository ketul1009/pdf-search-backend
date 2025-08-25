# document_service.py
"""
Service for processing uploaded documents and managing the FAISS search index.
"""

import os
import json
import uuid
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pypdf
from typing import List, Dict, Any

class DocumentProcessor:
    def __init__(self, index_path: str = "pdf_index.faiss", metadata_path: str = "metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load existing index and metadata if they exist
        self.index, self.metadata = self._load_existing_index()
    
    def _load_existing_index(self):
        """Load existing FAISS index and metadata, or create new ones."""
        try:
            index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Loaded existing index with {len(metadata)} chunks")
        except FileNotFoundError:
            print("📝 Creating new FAISS index and metadata")
            # Create new index with default dimension (384 for all-MiniLM-L6-v2)
            dimension = 384
            index = faiss.IndexFlatL2(dimension)
            metadata = []
        
        return index, metadata
    
    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split text into chunks with overlap."""
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
    
    def process_document(self, file_path: str, document_id: int, original_name: str) -> Dict[str, Any]:
        """Process a document file (PDF or TXT) and add it to the search index."""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found at '{file_path}'")
        
        print(f"Processing document: {original_name}...")
        
        # Extract text based on file type
        all_chunks = []
        new_metadata = []
        
        try:
            if original_name.lower().endswith('.pdf'):
                # Process PDF file
                reader = pypdf.PdfReader(file_path)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        page_chunks = self._chunk_text(text)
                        for chunk_index, chunk in enumerate(page_chunks):
                            all_chunks.append(chunk)
                            new_metadata.append({
                                "document_id": document_id,
                                "pdf_name": original_name,
                                "page": page_num,
                                "chunk_index": chunk_index,
                                "content": chunk
                            })
            elif original_name.lower().endswith('.txt'):
                # Process text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                if text:
                    chunks = self._chunk_text(text)
                    for chunk_index, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        new_metadata.append({
                            "document_id": document_id,
                            "pdf_name": original_name,
                            "page": 1,  # Text files are treated as single page
                            "chunk_index": chunk_index,
                            "content": chunk
                        })
            
            if not all_chunks:
                raise ValueError("No text could be extracted from the document.")
                
        except Exception as e:
            raise Exception(f"Error reading or processing document: {e}")
        
        # Generate embeddings for new chunks
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.model.encode(all_chunks, show_progress_bar=True, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add new embeddings to existing index
        self.index.add(embeddings)
        
        # Add new metadata
        self.metadata.extend(new_metadata)
        
        # Save updated index and metadata
        self._save_index()
        
        print(f"✅ Successfully processed {len(all_chunks)} chunks from {original_name}")
        
        return {
            "chunks_processed": len(all_chunks),
            "pages_processed": len(set(item["page"] for item in new_metadata)),
            "document_id": document_id
        }
    
    def _save_index(self):
        """Save the current FAISS index and metadata."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)
        print(f"💾 Saved index with {len(self.metadata)} total chunks")
    
    def search(self, query: str, document_id: int = None, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks, optionally scoped to a specific document."""
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0].astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        # Perform similarity search
        distances, indices = self.index.search(query_embedding, k)
        
        # Filter results by document if specified
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                chunk_info = self.metadata[idx]
                
                # If document_id is specified, only return chunks from that document
                if document_id is None or chunk_info.get("document_id") == document_id:
                    results.append({
                        "content": chunk_info["content"],
                        "pdf_name": chunk_info["pdf_name"],
                        "page": chunk_info["page"],
                        "distance": float(distances[0][i])
                    })
        
        return results
