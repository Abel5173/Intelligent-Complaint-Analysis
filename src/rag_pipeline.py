#!/usr/bin/env python3
"""
Task 3: RAG Pipeline Script

This script implements the Retrieval-Augmented Generation (RAG) pipeline
for intelligent complaint analysis.
"""

import pandas as pd
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import os
import logging
from typing import List, Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintRAGPipeline:
    """RAG pipeline for complaint analysis."""
    
    def __init__(self, vector_store_path: str = "../vector_store/complaint_embeddings",
                 model_name: str = "gpt-3.5-turbo"):
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self._load_components()
    
    def _load_components(self):
        """Load vector store and language model."""
        try:
            # Load vector store
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.load_local(self.vector_store_path, embeddings)
            logger.info("Vector store loaded successfully")
            
            # Initialize language model
            self.llm = OpenAI(model_name=self.model_name, temperature=0.1)
            logger.info(f"Language model {self.model_name} initialized")
            
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise
    
    def create_prompt_template(self) -> PromptTemplate:
        """Create a custom prompt template for complaint analysis."""
        
        template = """You are an expert financial complaint analyst. Use the following context to answer the question about consumer financial complaints.

Context: {context}

Question: {question}

Instructions:
1. Analyze the complaint context carefully
2. Provide a clear, informative answer
3. If the context doesn't contain enough information, say so
4. Focus on the specific financial product or service mentioned
5. Be helpful and professional in your response

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def setup_qa_chain(self, k: int = 4) -> RetrievalQA:
        """Set up the question-answering chain."""
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create prompt template
        prompt = self.create_prompt_template()
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logger.info(f"QA chain set up with k={k} retrieved documents")
        return self.qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: The question to ask about complaints
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            self.setup_qa_chain()
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "question": question
            }
        except Exception as e:
            logger.error(f"Error querying RAG pipeline: {e}")
            return {
                "answer": f"Error processing query: {e}",
                "source_documents": [],
                "question": question
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of results for each question
        """
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results
    
    def evaluate_retrieval(self, test_questions: List[str], 
                          expected_products: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            test_questions: List of test questions
            expected_products: List of expected product categories
            
        Returns:
            Dictionary with evaluation metrics
        """
        correct_retrievals = 0
        total_questions = len(test_questions)
        
        for i, question in enumerate(test_questions):
            result = self.query(question)
            retrieved_products = set()
            
            # Extract products from retrieved documents
            for doc in result["source_documents"]:
                if hasattr(doc, 'metadata') and 'product' in doc.metadata:
                    retrieved_products.add(doc.metadata['product'])
            
            # Check if expected product is in retrieved products
            if expected_products[i] in retrieved_products:
                correct_retrievals += 1
        
        accuracy = correct_retrievals / total_questions if total_questions > 0 else 0
        
        return {
            "retrieval_accuracy": accuracy,
            "correct_retrievals": correct_retrievals,
            "total_questions": total_questions
        }

def main():
    """Main function to test the RAG pipeline."""
    
    # Initialize RAG pipeline
    logger.info("Initializing RAG pipeline...")
    rag_pipeline = ComplaintRAGPipeline()
    
    # Test questions
    test_questions = [
        "What are the most common issues with credit card complaints?",
        "How do mortgage complaints typically get resolved?",
        "What are the main problems with student loan servicing?"
    ]
    
    # Test the pipeline
    logger.info("Testing RAG pipeline...")
    for question in test_questions:
        result = rag_pipeline.query(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {len(result['source_documents'])} documents retrieved")

if __name__ == "__main__":
    main() 