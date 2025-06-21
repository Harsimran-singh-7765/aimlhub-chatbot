import os
from dotenv import load_dotenv
load_dotenv()
print("🔐 Gemini Key Loaded:", os.getenv("GEMINI_API_KEY"))

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Vectorstore caching
vectorstore = None

# 🧠 System prompt for Gemini LLM
SYSTEM_PROMPT = f"""
You are AIML Hub's virtual assistant. Respond in a friendly, helpful, and informative way.
If you don't know the answer, suggest the user check with the AIML Hub team.
Avoid guessing. Always sound confident and human-like. Keep answers short and clear.



Context: {{context}}

Question: {{question}}
"""

prompt_template = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question"]
)

# 🔍 Load all .txt files from /data directory
def load_all_documents(directory: str = "chatbot/data") -> list:
    print(f"📂 Loading all .txt files from: {directory}")
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()

def clean_response(text: str) -> str:
    return text.replace("***", "").replace("**", "").strip()

# 📦 Vectorstore initialization
def load_vectorstore():
    global vectorstore

    print("📄 Loading documents...")
    docs = load_all_documents()

    print(f"📚 Loaded {len(docs)} documents")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print(f"✂️ Split into {len(chunks)} chunks")

    print("🧠 Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    print("📦 Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vectorstore ready.")

# 🤖 Main function to get chatbot response
def get_chatbot_response(query: str) -> str:
    global vectorstore
    print("💬 Incoming query:", query)

    try:
        if not vectorstore:
            load_vectorstore()

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        print("🤖 Initializing Gemini...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        print("🔗 Creating RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )

        print("⚙️ Running QA chain...")
        response = qa_chain.run(query)
        cleaned = clean_response(response)

        print("✅ Response:", cleaned)
        return cleaned

    except Exception as e:
        print("🔥 Error in get_chatbot_response:", str(e))
        return "⚠️ Something went wrong on the backend. Please try again later."
