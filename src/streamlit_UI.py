import streamlit as st
from src.pdf_upload import PDFProcessor
from src.vector_db import VectorStore
from src.qa_retrieval import GeminiQABot

class StreamlitApp:
    def __init__(self):
        st.set_page_config(page_title="Doc_Chat App")
        st.title("📄 Doc_Chat App")

    def run(self):
        # Sidebar for PDF upload
        with st.sidebar:
            st.header("Upload PDF")
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

            if uploaded_file is not None:
                with st.spinner("Processing PDF..."):
                    with open("temp_uploaded.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    processor = PDFProcessor("temp_uploaded.pdf")
                    text_chunks = processor.load_and_split()

                    vector_handler = VectorStore(text_chunks)
                    docsearch = vector_handler.build_faiss_index()

                    st.session_state.qa_bot = GeminiQABot(docsearch)
                    st.success("✅ Document processed.")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            if "qa_bot" in st.session_state:
                with st.spinner("Getting response..."):
                    result = st.session_state.qa_bot.ask(prompt)
                    response = result["result"]
                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("Please upload a PDF first.")
