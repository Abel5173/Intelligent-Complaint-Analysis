import logging
import os
import torch
import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === Logging setup ===
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Constants ===
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
BATCH_SIZE = 256
DATA_PATH = 'data/filtered_complaints.csv'
VECTOR_STORE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'vector_store/complaint_embeddings_mini'))

# === Embedding setup ===
embedder = HuggingFaceEmbeddings(
    model_name='thenlper/gte-small',
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

# === Text chunking ===


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if not text or not isinstance(text, str):
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# === Load and clean data ===


def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        column_map = {
            'Complaint ID': 'complaint_id',
            'Date received': 'date_received',
            'Product': 'product',
            'Issue': 'issue',
            'Company': 'company',
            'Consumer complaint narrative': 'narrative',
            'Sub-issue': 'sub_issue',
            'State': 'state',
            'Tags': 'tags',
            'Submitted via': 'submitted_via'
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        required = ['complaint_id', 'date_received', 'product', 'issue', 'company', 'narrative']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df.dropna(subset=['narrative'])
        logger.info(f"✅ Loaded {len(df)} complaints with narratives")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise

# === Convert complaints to LangChain Documents ===


def complaints_to_documents(df):
    documents = []
    for idx, row in df.iterrows():
        try:
            complaint_id = str(row['complaint_id'])
            narrative = str(row['narrative'])
            chunks = chunk_text(narrative)
            for i, chunk in enumerate(chunks):
                metadata = {
                    'complaint_id': complaint_id,
                    'chunk_index': i,
                    'product': row.get('product', ''),
                    'date_received': str(row.get('date_received', '')),
                    'issue': str(row.get('issue', '')),
                    'sub_issue': str(row.get('sub_issue', '')),
                    'company': str(row.get('company', '')),
                    'state': str(row.get('state', '')),
                    'tags': str(row.get('tags', '')),
                    'submitted_via': str(row.get('submitted_via', ''))
                }
                doc = Document(page_content=chunk, metadata=metadata)
                documents.append(doc)
        except Exception as e:
            logger.warning(f"⚠️ Skipping row due to error: {e}")
    logger.info(f"✅ Created {len(documents)} text chunks")
    return documents

# === Build and save vector store ===


def build_vector_store(documents):
    try:
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, "chroma.sqlite3")):
            logger.info("🗑️ Deleting existing vector store")
            for file in os.listdir(VECTOR_STORE_PATH):
                os.remove(os.path.join(VECTOR_STORE_PATH, file))
        logger.info("🚀 Embedding and storing documents...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedder,
            persist_directory=VECTOR_STORE_PATH
        )
        vectorstore.persist()
        logger.info(f"✅ Vector store saved with {len(documents)} documents")
        return vectorstore
    except Exception as e:
        logger.error(f"❌ Failed to build vector store: {e}")
        raise


# === Main pipeline ===
if __name__ == "__main__":
    try:
        df = load_data()
        df = df.sample(n=10000, random_state=42)
        documents = complaints_to_documents(df)
        build_vector_store(documents)
    except Exception as e:
        logger.error(f"🚨 Pipeline failed: {e}")
