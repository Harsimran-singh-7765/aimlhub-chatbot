import os
from dotenv import load_dotenv
load_dotenv()
print("🔐 Gemini Key Loaded:", os.getenv("GEMINI_API_KEY"))

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

def load_vectorstore():
    filepath = "chatbot/data/knowledge.txt"
    if not os.path.exists(filepath):
        print("❌ Knowledge file not found at:", filepath)
        raise FileNotFoundError(f"Knowledge file not found at {filepath}")

    print("📄 Loading knowledge file...")
    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()

    print(f"📚 Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print(f"✂️ Split into {len(chunks)} chunks")

    print("🔍 Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
    )


    print("📦 Creating vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("✅ Vectorstore loaded successfully")
    return vectorstore

def get_chatbot_response(query: str) -> str:
    print("💬 Incoming user query:", query)

    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        print("🤖 Initializing Gemini LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        print("🔗 Setting up QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        print("⚙️ Running QA chain...")
        response = qa_chain.run(query)

        print("✅ Final response:", response)
        return response

    except Exception as e:
        print("🔥 ERROR in get_chatbot_response():", str(e))
        return "⚠️ Something went wrong. Check backend logs."
