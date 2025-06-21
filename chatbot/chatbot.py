import os
from dotenv import load_dotenv
load_dotenv()
print("🔐 Gemini Key Loaded:", os.getenv("GEMINI_API_KEY"))

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Global vectorstore caching
vectorstore = None

# 🧠 Custom System Prompt
SYSTEM_PROMPT = """
You are AIML Hub's virtual assistant. Respond in a friendly, helpful, and informative way. 
If you don't know the answer, suggest the user check with the AIML Hub team. 
Avoid guessing. Always sound confident and human-like. Try to keep answers short and clear.
Context: {context}

Question: {question}
"""

prompt_template = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question"]
)

def load_vectorstore():
    global vectorstore
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

    print("✅ Vectorstore initialized")

def get_chatbot_response(query: str) -> str:
    global vectorstore

    print("💬 Incoming user query:", query)

    try:
        if not vectorstore:
            load_vectorstore()

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        print("🤖 Initializing Gemini LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        print("🔗 Setting up RetrievalQA with custom prompt...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )

        print("⚙️ Running QA chain...")
        response = qa_chain.run(query)

        print("✅ Final response:", response)
        return response

    except Exception as e:
        print("🔥 ERROR in get_chatbot_response():", str(e))
        return "⚠️ Something went wrong. Check backend logs."
