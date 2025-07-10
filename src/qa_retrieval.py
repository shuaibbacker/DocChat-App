import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

class GeminiQABot:
    def __init__(self, vector_store):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.5,
            max_tokens=2048,
            gemini_api_key=os.environ["GEMINI_API_KEY"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True
        )

    def ask(self, question):
        return self.qa_chain.invoke({"query": question})
