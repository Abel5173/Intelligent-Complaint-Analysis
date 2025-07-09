import logging
import re
import pandas as pd
from rag_pipeline import rag_pipeline
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_groq_response(response: str) -> str:
    """
    Clean the Groq LLM output by removing internal reasoning, think tags,
    and verbose chunk listings, ensuring a concise, user-friendly answer.
    """
    # Remove <think> tags and reasoning steps
    cleaned = re.sub(r'<think>.*?</think>|<think>.*?(?=\n\n|$)',
                     '', response, flags=re.DOTALL)
    # Remove verbose chunk listings (e.g., "First, the complaint from...")
    cleaned = re.sub(
        r'(First|Next|Then|The \w+ entry).*?(?=\n\n|$)', '', cleaned, flags=re.DOTALL)
    # Remove trailing ellipses or incomplete sentences
    cleaned = re.sub(r'\.\.\.+$|[^\.\!\?]+$', '', cleaned.strip())
    return cleaned.strip() or response.strip()


# Custom CSS for improved UI
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stTextInput > div > div > input { border-radius: 5px; padding: 10px; }
    .stButton > button { border-radius: 5px; padding: 8px 16px; }
    .stExpander { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 5px; }
    .stDataFrame { margin-top: 10px; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    .stTabs { margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.set_page_config(page_title="CreditTrust Complaint Analysis", layout="wide")
st.title("💡 CreditTrust Complaint Analysis")
st.markdown("Ask questions about customer complaints in the financial domain. Answers are generated using complaint excerpts and metadata from CFPB data, with sources displayed for transparency.")

# Sidebar with instructions and example questions
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
        - Enter a question in the 'Query' tab (e.g., "What issues arise with Money transfers?").
        - Click **Submit** to get a streamed answer and view source complaint chunks.
        - Use the **Clear** button to reset the conversation.
        - Select a source chunk from the dropdown or view the metadata table.
        - Check the 'History' tab to review past questions and answers.
        **Example Questions**:
        - Why are people unhappy with BNPL?
        - What are common Credit card issues?
        - Are there complaints from older consumers?
        - What issues arise with Money transfers?
    """)

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""
if 'current_chunks' not in st.session_state:
    st.session_state.current_chunks = []

# Check vector store status
try:
    from rag_pipeline import vectorstore
    doc_count = vectorstore._collection.count() if vectorstore else 0
    if not vectorstore:
        st.warning("The vector store is missing. Please run `src/chunking_embedding.py` to create and populate `complaints_collection` with data from `data/filtered_complaints.csv`.")
    elif doc_count == 0:
        st.warning("The vector store is empty. Please run `src/chunking_embedding.py` to populate it with data from `data/filtered_complaints.csv`.")
except Exception as e:
    st.warning(
        f"Error accessing vector store: {e}. Please run `src/chunking_embedding.py` to create and populate `complaints_collection`.")

# Create tabs for Query and History
query_tab, history_tab = st.tabs(["Query", "History"])

# Query tab
with query_tab:
    # Input form
    with st.form(key="query_form"):
        question = st.text_input(
            label="Enter your question about customer complaints",
            placeholder="e.g., What issues arise with Money transfers?"
        )
        # Adjust column widths to push Clear to the right
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            submit_button = st.form_submit_button("Submit")
        with col3:
            clear_button = st.form_submit_button("Clear")

    # Handle submit
    if submit_button and question:
        try:
            with st.spinner("🔍 Generating answer..."):
                result = rag_pipeline(question, k=5, stream=True)
                st.session_state.current_answer = ""
                answer_container = st.empty()
                with answer_container:
                    st.markdown("### 📝 Answer")
                    for token in result["answer"]:
                        st.session_state.current_answer += token
                        st.markdown(clean_groq_response(
                            st.session_state.current_answer))
                st.session_state.current_chunks = result["retrieved_chunks"]
                st.session_state.conversation.append(
                    (question, clean_groq_response(st.session_state.current_answer)))
        except Exception as e:
            st.error(f"Error processing question: {e}")
            st.session_state.current_answer = f"Error: {e}"
            st.session_state.current_chunks = []

    # Display sources
    if st.session_state.current_chunks:
        st.markdown("### 📚 Source Complaint Chunks")
        # Dropdown for selecting a chunk
        chunk_options = [f"Chunk {i+1} (Product: {chunk[1].get('product', 'N/A')}, Issue: {chunk[1].get('issue', 'N/A')})" for i,
                         chunk in enumerate(st.session_state.current_chunks)]
        selected_chunk = st.selectbox("Select a source chunk to view:", [
                                      "Select a chunk"] + chunk_options)
        if selected_chunk != "Select a chunk":
            chunk_idx = int(selected_chunk.split(" ")[1].split(" ")[0]) - 1
            chunk_text, chunk_metadata, chunk_score = st.session_state.current_chunks[
                chunk_idx]
            st.markdown(f"**Chunk Text**: {chunk_text}")
            st.markdown(
                f"**Metadata**: Product: {chunk_metadata.get('product', 'N/A')}, Issue: {chunk_metadata.get('issue', 'N/A')}, Date: {chunk_metadata.get('date_received', 'N/A')}, Company: {chunk_metadata.get('company', 'N/A')}, Score: {chunk_score:.3f}")

        # Table of metadata
        st.markdown("#### Source Metadata Table")
        source_data = [
            {
                "Chunk": f"Chunk {i+1}",
                "Product": chunk[1].get("product", "N/A"),
                "Issue": chunk[1].get("issue", "N/A"),
                "Date": chunk[1].get("date_received", "N/A"),
                "Company": chunk[1].get("company", "N/A"),
                "State": chunk[1].get("state", "N/A"),
                "Tags": chunk[1].get("tags", "N/A"),
                "Score": f"{chunk[2]:.3f}"
            }
            for i, chunk in enumerate(st.session_state.current_chunks)
        ]
        st.dataframe(pd.DataFrame(source_data), use_container_width=True)

    # Instructions for troubleshooting
    st.markdown("""
    **Note**: If no answers are generated, ensure `src/chunking_embedding.py` has been run to populate the vector store with data from `data/filtered_complaints.csv`. Check logs for errors if issues persist.
    """)

# History tab
with history_tab:
    st.markdown("### 📜 Conversation History")
    if st.session_state.conversation:
        for i, (question, answer) in enumerate(st.session_state.conversation):
            with st.container():
                st.markdown(f"**Question {i+1}**: {question}")
                st.markdown(f"**Answer**: {answer}")
                st.markdown("---")
    else:
        st.markdown(
            "No conversation history yet. Ask a question in the Query tab to start!")
