from dotenv import load_dotenv
from groq import Groq
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re
import pandas as pd
import torch
import os
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
VECTOR_STORE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'vector_store/complaint_embeddings_mini'))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY not set in environment variables")

# Initialize embeddings
embedder = HuggingFaceEmbeddings(
    model_name='thenlper/gte-small',
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

# Load existing vector store
try:
    vectorstore = Chroma(
        embedding_function=embedder,
        persist_directory=VECTOR_STORE_PATH
    )
    doc_count = vectorstore._collection.count()
    logger.info(f"✅ Loaded vector store with {doc_count} documents")
    if doc_count == 0:
        logger.warning(
            "⚠️ Vector store is empty. Please ensure the vector store is correctly populated.")
except Exception as e:
    logger.error(f"❌ Failed to load vector store: {e}")
    raise

# Initialize Groq client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    logger.error(f"❌ Error initializing Groq client: {e}")
    raise

# Prompt template for Groq LLM
system_prompt = (
    "You are a financial complaint analyst for CreditTrust Financial. "
    "Answer the user's question concisely using only the provided complaint excerpts and metadata. "
    "Cite specific examples with metadata (e.g., company, date, issue) to support your answer. "
    "Do not include reasoning steps, <think> tags, or list all chunks. "
    "If the context lacks sufficient information, state so clearly."
)

prompt_template = """
Context:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

bnpl_keywords = ["bnpl", "buy now pay later",
                 "quadpay", "affirm", "afterpay", "zip"]


def is_relevant(text):
    return any(kw in text.lower() for kw in bnpl_keywords)


def trim_incomplete_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:-1]) if not text.endswith(('.', '!', '?')) else text


def retrieve_relevant_chunks(question, k=5):
    try:
        docs_and_scores = vectorstore.similarity_search_with_score(
            question, k=k * 2)
        chunks = [
            (doc.page_content, doc.metadata, score)
            for doc, score in docs_and_scores
        ]
        filtered_chunks = [
            c for c in chunks if is_relevant(c[0])] or chunks[:k]
        logger.info(
            f"Retrieved {len(filtered_chunks)} relevant chunks for question: '{question}'")
        return filtered_chunks[:k]
    except Exception as e:
        logger.error(f"Error retrieving chunks for question '{question}': {e}")
        return []


def rag_pipeline(question, k=5, stream=False):
    try:
        if vectorstore._collection.count() == 0:
            return {
                "question": question,
                "answer": "No relevant complaint data found in the vector store. Please populate the vector store first.",
                "retrieved_chunks": []
            }

        chunks = retrieve_relevant_chunks(question, k=k)
        if not chunks:
            return {
                "question": question,
                "answer": "No relevant complaint data found for this question.",
                "retrieved_chunks": []
            }

        context = "\n\n".join([
            f"(Product: {chunk[1].get('product', 'N/A')}, Issue: {chunk[1].get('issue', 'N/A')}, "
            f"Date: {chunk[1].get('date_received', 'N/A')}, Company: {chunk[1].get('company', 'N/A')}):\n{chunk[0]}"
            for chunk in chunks
        ])
        formatted_prompt = prompt.format(context=context, question=question)

        if stream:
            def stream_response():
                for chunk in groq_client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.6,
                    max_completion_tokens=500,
                    top_p=0.95,
                    stream=True
                ):
                    content = chunk.choices[0].delta.content or ""
                    yield content
            return {
                "answer": stream_response(),
                "retrieved_chunks": chunks
            }
        else:
            completion = groq_client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.6,
                max_completion_tokens=500,
                top_p=0.95,
                stream=False
            )
            raw_answer = completion.choices[0].message.content.strip()
            cleaned_answer = trim_incomplete_sentences(raw_answer)
            return {
                "question": question,
                "answer": cleaned_answer,
                "retrieved_chunks": chunks
            }
    except Exception as e:
        logger.error(f"Error processing question '{question}': {e}")
        return {
            "question": question,
            "answer": f"Error: Failed to generate answer ({str(e)})",
            "retrieved_chunks": []
        }


# Evaluation
if __name__ == "__main__":
    questions = [
        "Why are people unhappy with BNPL?",
        "What are common Credit card issues?",
        "Are there Savings account complaints?",
        "What issues arise with Money transfers?",
        "Why do customers complain about Personal loans?",
        "What are recent complaints in California?",
        "Are there complaints from older consumers?",
        "What are customer service issues with BNPL?"
    ]

    evaluation_results = []
    for question in questions:
        result = rag_pipeline(question)
        top_chunks = result['retrieved_chunks'][:2]
        evaluation_results.append({
            "question": question,
            "answer": result['answer'],
            "sources": [
                f"{chunk[0][:100]}... (Product: {chunk[1].get('product', 'N/A')}, "
                f"Issue: {chunk[1].get('issue', 'N/A')}, Date: {chunk[1].get('date_received', 'N/A')}, "
                f"Company: {chunk[1].get('company', 'N/A')})"
                for chunk in top_chunks
            ] if top_chunks else [],
            "quality_score": 0,
            "comments": "Pending manual evaluation"
        })
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print("Top 2 Retrieved Chunks:")
        if top_chunks:
            for i, chunk in enumerate(top_chunks, 1):
                print(
                    f"Chunk {i}: {chunk[0][:100]}... (Metadata: {chunk[1]}, Score: {chunk[2]:.4f})"
                )
        else:
            print("No chunks retrieved.")

    pd.DataFrame(evaluation_results).to_csv(
        'notebooks/evaluation_results.csv', index=False)
    logger.info("✅ Evaluation results saved to notebooks/evaluation_results.csv")
