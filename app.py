import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np
import faiss


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Free RAG PDF Q&A App",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Free RAG PDF Q&A App")
st.write("Upload a PDF and ask questions from it using Gemini + FAISS + free embeddings.")


# -----------------------------
# Gemini Client
# -----------------------------
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI_API_KEY is missing. Please add it in Streamlit secrets.")
        st.stop()

    return genai.Client(api_key=api_key)


client = get_gemini_client()

# You can change this if needed.
GEMINI_MODEL = "gemini-2.5-flash"


# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text += f"\n\n--- Page {page_number} ---\n"
            text += page_text

    return text


# -----------------------------
# Split text into chunks
# -----------------------------
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


# -----------------------------
# Create embeddings
# -----------------------------
def create_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")


# -----------------------------
# Build FAISS index
# -----------------------------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# -----------------------------
# Retrieve relevant chunks
# -----------------------------
def retrieve_relevant_chunks(question, chunks, index, top_k=3):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    relevant_chunks = []

    for idx in indices[0]:
        if idx < len(chunks):
            relevant_chunks.append(chunks[idx])

    return relevant_chunks


# -----------------------------
# Generate answer using Gemini
# -----------------------------
def generate_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the context provided below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "I could not find this information in the uploaded document."
3. Keep the answer clear and simple.
4. If possible, mention which part of the context supports the answer.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"""
Gemini API error occurred.

Most common reasons:
1. Gemini API key is incorrect.
2. Gemini API key is not added in Streamlit Cloud secrets.
3. The selected model is not available for your API key.
4. Your free quota is exhausted.
5. Gemini API is not enabled for your Google project.

Current model used:
{GEMINI_MODEL}

Original error:
{str(e)}
"""


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Project Info")
    st.write("This is a simple RAG app.")

    st.markdown("""
    **Flow:**
    1. Upload PDF  
    2. Extract text  
    3. Create chunks  
    4. Generate embeddings  
    5. Store vectors in FAISS  
    6. Retrieve relevant chunks  
    7. Ask Gemini to answer  
    """)

    st.markdown("---")
    st.write(f"LLM Model: `{GEMINI_MODEL}`")
    st.write("Embedding Model: `all-MiniLM-L6-v2`")


# -----------------------------
# Streamlit UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF file",
    type=["pdf"]
)

if uploaded_file:
    st.success("PDF uploaded successfully.")

    if st.button("Process Document"):
        with st.spinner("Reading PDF and creating embeddings..."):
            text = extract_text_from_pdf(uploaded_file)

            if not text.strip():
                st.error(
                    "No text found in this PDF. It may be a scanned image PDF."
                )
            else:
                chunks = chunk_text(text)
                embeddings = create_embeddings(chunks)
                index = build_faiss_index(embeddings)

                st.session_state["text"] = text
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["document_ready"] = True

                st.success(
                    f"Document processed successfully. Total chunks: {len(chunks)}"
                )

                with st.expander("Preview extracted text"):
                    st.write(text[:3000])


if st.session_state.get("document_ready"):
    st.subheader("Ask a question from your document")

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Searching document and generating answer..."):
                relevant_chunks = retrieve_relevant_chunks(
                    question=question,
                    chunks=st.session_state["chunks"],
                    index=st.session_state["index"],
                    top_k=3
                )

                answer = generate_answer(question, relevant_chunks)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("View retrieved chunks"):
                    for i, chunk in enumerate(relevant_chunks, start=1):
                        st.markdown(f"### Chunk {i}")
                        st.write(chunk)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload and process a PDF first, then ask questions from it.")