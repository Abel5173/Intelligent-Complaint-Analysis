#!/usr/bin/env python3
"""
Task 4: Gradio/Streamlit App for Intelligent Complaint Analysis

This script creates a web application for the complaint analysis system.
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import os
import sys

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import ComplaintRAGPipeline

class ComplaintAnalysisApp:
    """Gradio app for complaint analysis."""
    
    def __init__(self):
        self.rag_pipeline = None
        self.complaints_data = None
        self._load_data()
    
    def _load_data(self):
        """Load complaints data and initialize RAG pipeline."""
        try:
            # Load filtered complaints data
            self.complaints_data = pd.read_csv("../data/filtered_complaints.csv")
            
            # Initialize RAG pipeline
            self.rag_pipeline = ComplaintRAGPipeline()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.complaints_data = pd.DataFrame()
    
    def analyze_complaint(self, question: str) -> str:
        """Analyze a complaint-related question using RAG."""
        if not self.rag_pipeline:
            return "Error: RAG pipeline not initialized. Please check the vector store."
        
        try:
            result = self.rag_pipeline.query(question)
            return result["answer"]
        except Exception as e:
            return f"Error analyzing complaint: {e}"
    
    def get_product_distribution(self) -> go.Figure:
        """Create product distribution visualization."""
        if self.complaints_data.empty:
            return go.Figure()
        
        product_counts = self.complaints_data['Product'].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=product_counts.values,
                y=product_counts.index,
                orientation='h',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="Top 10 Product Categories by Complaint Count",
            xaxis_title="Number of Complaints",
            yaxis_title="Product Category",
            height=500
        )
        
        return fig
    
    def get_issue_distribution(self) -> go.Figure:
        """Create issue distribution visualization."""
        if self.complaints_data.empty:
            return go.Figure()
        
        issue_counts = self.complaints_data['Issue'].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=issue_counts.values,
                y=issue_counts.index,
                orientation='h',
                marker_color='lightcoral'
            )
        ])
        
        fig.update_layout(
            title="Top 10 Issue Types by Complaint Count",
            xaxis_title="Number of Complaints",
            yaxis_title="Issue Type",
            height=500
        )
        
        return fig
    
    def get_timeline_analysis(self) -> go.Figure:
        """Create timeline analysis visualization."""
        if self.complaints_data.empty:
            return go.Figure()
        
        # Convert date column and group by month
        self.complaints_data['Date received'] = pd.to_datetime(self.complaints_data['Date received'])
        monthly_counts = self.complaints_data.groupby(
            self.complaints_data['Date received'].dt.to_period('M')
        ).size()
        
        fig = go.Figure(data=[
            go.Scatter(
                x=monthly_counts.index.astype(str),
                y=monthly_counts.values,
                mode='lines+markers',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            )
        ])
        
        fig.update_layout(
            title="Complaint Volume Over Time",
            xaxis_title="Month",
            yaxis_title="Number of Complaints",
            height=400
        )
        
        return fig
    
    def get_company_analysis(self) -> go.Figure:
        """Create company analysis visualization."""
        if self.complaints_data.empty:
            return go.Figure()
        
        company_counts = self.complaints_data['Company'].value_counts().head(15)
        
        fig = go.Figure(data=[
            go.Bar(
                x=company_counts.values,
                y=company_counts.index,
                orientation='h',
                marker_color='lightgreen'
            )
        ])
        
        fig.update_layout(
            title="Top 15 Companies by Complaint Count",
            xaxis_title="Number of Complaints",
            yaxis_title="Company",
            height=600
        )
        
        return fig

def create_app():
    """Create and return the Gradio app interface."""
    
    app_instance = ComplaintAnalysisApp()
    
    with gr.Blocks(
        title="Intelligent Complaint Analysis System",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🏦 Intelligent Complaint Analysis System
        
        This application provides intelligent analysis of consumer financial complaints using advanced AI techniques.
        """)
        
        with gr.Tabs():
            
            # Tab 1: RAG Analysis
            with gr.TabItem("🤖 AI Analysis"):
                gr.Markdown("""
                ### Ask questions about consumer complaints
                
                Use natural language to ask questions about financial complaints, product issues, 
                company responses, and more. The AI will analyze the complaint database and provide 
                intelligent insights.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Ask a question about complaints",
                            placeholder="e.g., What are the most common credit card issues?",
                            lines=3
                        )
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column(scale=3):
                        answer_output = gr.Textbox(
                            label="AI Analysis",
                            lines=8,
                            interactive=False
                        )
                
                analyze_btn.click(
                    fn=app_instance.analyze_complaint,
                    inputs=question_input,
                    outputs=answer_output
                )
            
            # Tab 2: Data Visualizations
            with gr.TabItem("📊 Data Insights"):
                gr.Markdown("""
                ### Explore complaint data through interactive visualizations
                
                Analyze patterns, trends, and distributions in the complaint dataset.
                """)
                
                with gr.Row():
                    with gr.Column():
                        product_plot = gr.Plot(label="Product Distribution")
                        issue_plot = gr.Plot(label="Issue Distribution")
                    
                    with gr.Column():
                        timeline_plot = gr.Plot(label="Timeline Analysis")
                        company_plot = gr.Plot(label="Company Analysis")
                
                # Load visualizations
                product_plot.value = app_instance.get_product_distribution()
                issue_plot.value = app_instance.get_issue_distribution()
                timeline_plot.value = app_instance.get_timeline_analysis()
                company_plot.value = app_instance.get_company_analysis()
            
            # Tab 3: Quick Stats
            with gr.TabItem("📈 Quick Statistics"):
                gr.Markdown("""
                ### Key statistics about the complaint dataset
                """)
                
                if not app_instance.complaints_data.empty:
                    total_complaints = len(app_instance.complaints_data)
                    unique_products = app_instance.complaints_data['Product'].nunique()
                    unique_companies = app_instance.complaints_data['Company'].nunique()
                    date_range = f"{app_instance.complaints_data['Date received'].min()} to {app_instance.complaints_data['Date received'].max()}"
                    
                    gr.Markdown(f"""
                    - **Total Complaints**: {total_complaints:,}
                    - **Unique Products**: {unique_products}
                    - **Unique Companies**: {unique_companies}
                    - **Date Range**: {date_range}
                    """)
                else:
                    gr.Markdown("No data available. Please check the data files.")
            
            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This System
                
                This Intelligent Complaint Analysis System uses advanced AI techniques including:
                
                - **Retrieval-Augmented Generation (RAG)**: Combines information retrieval with text generation
                - **Vector Embeddings**: Converts text into numerical representations for similarity search
                - **Natural Language Processing**: Understands and responds to questions in natural language
                
                ### How it works:
                1. **Data Processing**: Raw complaints are cleaned and preprocessed
                2. **Text Chunking**: Long complaint narratives are split into manageable chunks
                3. **Embedding Generation**: Each chunk is converted to a vector embedding
                4. **Vector Storage**: Embeddings are stored in a searchable vector database
                5. **Query Processing**: User questions are processed and relevant chunks are retrieved
                6. **Answer Generation**: AI generates comprehensive answers based on retrieved context
                
                ### Technology Stack:
                - **LangChain**: Framework for building LLM applications
                - **OpenAI**: Language models and embeddings
                - **FAISS**: Vector similarity search
                - **Gradio**: Web interface
                - **Pandas/NumPy**: Data processing
                - **Plotly**: Interactive visualizations
                """)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    ) 