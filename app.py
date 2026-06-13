"""
PDF RAG QA Bot
--------------
A Retrieval-Augmented Generation chatbot that answers questions ONLY based on
an uploaded PDF. Out-of-scope questions are politely declined.

Stack:
- Streamlit (UI)
- Google Gemini (LLM + embeddings) - free tier
- FAISS (vector store)
- LangChain (orchestration)
"""

import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PDF RAG QA Bot", page_icon="📄", layout="centered")
st.title("📄 PDF Q&A Bot")
st.caption("Ask questions about the uploaded PDF. The bot will only answer from the document's content.")

# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------
# Priority: Streamlit secrets (for deployment) -> environment variable (local)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

if not GOOGLE_API_KEY:
    st.warning("⚠️ No Google API key found. Add it in `.streamlit/secrets.toml` or as an environment variable `GOOGLE_API_KEY`.")
    GOOGLE_API_KEY = st.text_input("Or paste your Gemini API key here (not stored):", type="password")

if not GOOGLE_API_KEY:
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------------------------------------------------------------------
# Prompt template - this is what enforces "answer only from the PDF"
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly based on
the provided context from a PDF document.

Rules:
1. Answer ONLY using information present in the context below.
2. If the answer is not contained in the context, respond EXACTLY with:
   "I'm designed to answer questions only about the uploaded document. This question appears to be outside its scope."
3. Do not use any outside knowledge, even if you know the answer.
4. Be concise and cite the relevant part of the document where possible.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# ---------------------------------------------------------------------------
# Caching: build the vector store once per uploaded file
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_qa_chain(file_bytes: bytes, file_name: str):
    """Load PDF, chunk, embed, build FAISS index, and return a QA chain."""
    # Save to a temp file because PyPDFLoader needs a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise ValueError("Could not extract any text from this PDF. It may be a scanned/image-only PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
        )
        return qa_chain, len(chunks)
    finally:
        os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# UI - file upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    with st.spinner("Processing PDF — chunking and building index..."):
        try:
            qa_chain, num_chunks = build_qa_chain(file_bytes, uploaded_file.name)
            st.success(f"✅ Indexed '{uploaded_file.name}' into {num_chunks} chunks. Ask away!")
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
            st.stop()

    # -------------------------------------------------------------------
    # Chat interface
    # -------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about this PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain.invoke({"query": question})
                    answer = result["result"]

                    st.markdown(answer)

                    # Show retrieved source chunks for transparency (collapsible)
                    with st.expander("📚 Sources used"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(f"**Chunk {i} (page {page}):**")
                            st.text(doc.page_content[:300] + "...")

                except Exception as e:
                    answer = f"Error: {e}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

else:
    st.info("👆 Upload a PDF to get started.")
    st.markdown("""
    **How this works:**
    1. Your PDF is split into overlapping text chunks
    2. Each chunk is converted into an embedding (vector) using Gemini
    3. When you ask a question, it's embedded too, and the most similar chunks are retrieved
    4. Gemini answers using ONLY those retrieved chunks
    5. If the answer isn't in the document, the bot tells you it's out of scope
    """)
