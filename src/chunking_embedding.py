#!/usr/bin/env python3
"""
Task 2: Text Chunking and Embedding Script

This script handles the text chunking, embedding generation, and vector store creation
for the CFPB complaints dataset.
"""

import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import logging
from typing import List, Dict, Any
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintChunker:
    """Handles text chunking for complaint narratives."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_complaints(self, complaints_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Chunk complaint narratives into smaller pieces.
        
        Args:
            complaints_df: DataFrame containing complaint data
            
        Returns:
            List of dictionaries containing chunked text and metadata
        """
        chunks = []
        
        for idx, row in tqdm(complaints_df.iterrows(), total=len(complaints_df), desc="Chunking complaints"):
            narrative = row.get('Consumer complaint narrative', '')
            if pd.isna(narrative) or narrative == '':
                continue
                
            # Split the narrative into chunks
            text_chunks = self.text_splitter.split_text(narrative)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                chunk_data = {
                    'text': chunk,
                    'complaint_id': row.get('Complaint ID', f'complaint_{idx}'),
                    'product': row.get('Product', 'Unknown'),
                    'issue': row.get('Issue', 'Unknown'),
                    'company': row.get('Company', 'Unknown'),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks)
                }
                chunks.append(chunk_data)
        
        logger.info(f"Created {len(chunks)} chunks from {len(complaints_df)} complaints")
        return chunks

class ComplaintEmbedder:
    """Handles embedding generation for complaint chunks."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=model_name)
    
    def create_vector_store(self, chunks: List[Dict[str, Any]], 
                          vector_store_path: str = "../vector_store/complaint_embeddings") -> FAISS:
        """
        Create and save vector store from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            vector_store_path: Path to save the vector store
            
        Returns:
            FAISS vector store object
        """
        # Extract texts and metadata
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [{k: v for k, v in chunk.items() if k != 'text'} for chunk in chunks]
        
        # Create vector store
        logger.info("Creating vector store...")
        vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        
        # Save vector store
        os.makedirs(vector_store_path, exist_ok=True)
        vector_store.save_local(vector_store_path)
        logger.info(f"Vector store saved to {vector_store_path}")
        
        return vector_store

def main():
    """Main function to run the chunking and embedding pipeline."""
    
    # Load filtered complaints data
    logger.info("Loading filtered complaints data...")
    df = pd.read_csv("../data/filtered_complaints.csv")
    logger.info(f"Loaded {len(df)} complaints")
    
    # Initialize chunker and embedder
    chunker = ComplaintChunker()
    embedder = ComplaintEmbedder()
    
    # Chunk the complaints
    logger.info("Chunking complaint narratives...")
    chunks = chunker.chunk_complaints(df)
    
    # Create vector store
    logger.info("Creating vector store...")
    vector_store = embedder.create_vector_store(chunks)
    
    logger.info("Chunking and embedding pipeline completed successfully!")

if __name__ == "__main__":
    main() 