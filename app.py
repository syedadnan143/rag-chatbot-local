import os
import tempfile
from typing import List

import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="RAG Chatbot (Local)", layout="wide")
st.title("📄🔎 RAG Chatbot (Local Embeddings)")

# -------- Helpers --------
def load_documents(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    for uf in uploaded_files:
        suffix = os.path.splitext(uf.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name

        if suffix == ".pdf":
            docs.extend(PyPDFLoader(tmp_path).load())
        elif suffix in [".txt", ".md"]:
            docs.extend(TextLoader(tmp_path, encoding="utf-8").load())
        else:
            st.warning(f"Unsupported file type: {uf.name} (use PDF/TXT/MD)")
    return docs

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

@st.cache_resource
def get_embeddings():
    # Downloads once, then cached locally
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)

def retrieve(vs: FAISS, question: str, k: int = 4) -> List[Document]:
    return vs.similarity_search(question, k=k)

def simple_grounded_answer(question: str, retrieved_docs: List[Document]) -> str:
    """
    No-LLM fallback answer:
    - shows the most relevant passages as the 'answer'
    """
    if not retrieved_docs:
        return "I don’t know based on the provided documents."

    answer = "Here are the most relevant passages I found:\n\n"
    for i, d in enumerate(retrieved_docs, start=1):
        src = d.metadata.get("source", "uploaded_doc")
        page = d.metadata.get("page", None)
        cite = f"[{i}] {os.path.basename(src)}" + (f" (page {page+1})" if isinstance(page, int) else "")
        snippet = d.page_content.strip()
        answer += f"{cite}\n{snippet}\n\n"
    return answer

# -------- UI --------
st.sidebar.header("1) Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/TXT/MD files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

build_btn = st.sidebar.button("2) Build / Rebuild Knowledge Base")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if build_btn:
    try:
        if not uploaded_files:
            st.sidebar.error("Upload at least one file first.")
        else:
            with st.spinner("Loading documents..."):
                docs = load_documents(uploaded_files)

            with st.spinner("Splitting into chunks..."):
                chunks = split_documents(docs)

            with st.spinner("Building vector index (FAISS + local embeddings)..."):
                st.session_state.vectorstore = build_vectorstore(chunks)

            st.sidebar.success(f"Indexed {len(chunks)} chunks ✅")
    except Exception as e:
        st.sidebar.error("Build failed. See details below.")
        st.exception(e)

st.header("3) Ask questions")
if st.session_state.vectorstore is None:
    st.info("Upload documents and click **Build / Rebuild Knowledge Base** to start.")
else:
    question = st.text_input("Ask a question about your uploaded docs:")
    top_k = st.slider("How many chunks to retrieve?", 2, 8, 4)

    if st.button("Ask"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Retrieving relevant passages..."):
                retrieved = retrieve(st.session_state.vectorstore, question, k=top_k)

            st.subheader("Answer")
            st.write(simple_grounded_answer(question, retrieved))

            with st.expander("See retrieved text (debug)"):
                for i, d in enumerate(retrieved, start=1):
                    st.markdown(f"**SOURCE {i}**")
                    st.write(d.page_content)
                    st.divider()
