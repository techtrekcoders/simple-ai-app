import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np
import faiss


st.set_page_config(
    page_title="Free RAG PDF Q&A App",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Free RAG PDF Q&A App")
st.write("Upload a PDF and ask questions from it using a free-tier LLM setup.")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

llm_model = genai.GenerativeModel("gemini-1.5-flash")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


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


def create_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_relevant_chunks(question, chunks, index, top_k=3):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    relevant_chunks = []
    for idx in indices[0]:
        relevant_chunks.append(chunks[idx])

    return relevant_chunks


def generate_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the context below.

If the answer is not available in the context, say:
"I could not find this information in the uploaded document."

Context:
{context}

Question:
{question}
"""

    response = llm_model.generate_content(prompt)
    return response.text


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
                st.error("No text found in this PDF. It may be a scanned/scanned-image PDF.")
            else:
                chunks = chunk_text(text)
                embeddings = create_embeddings(chunks)
                index = build_faiss_index(embeddings)

                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["document_ready"] = True

                st.success(f"Document processed successfully. Total chunks: {len(chunks)}")


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