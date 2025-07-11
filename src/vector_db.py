from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


class VectorStore:
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_faiss_index(self):
        return FAISS.from_documents(self.text_chunks, self.embeddings)
