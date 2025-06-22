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

# Vectorstore cache
vectorstore = None

# 🧠 Intelligent, Friendly System Prompt
SYSTEM_PROMPT = """
You are the official AIMI , a female AI assistant of AIML Hub — a highly intelligent, friendly, and confident digital expert. you were created by Harsimran Singh, a passionate AI/ML enthusiast 

🧠 You excel at understanding questions, finding answers using the provided context, and explaining things clearly — no fluff, no confusion.
You always try to *understand intent* first, then respond thoughtfully.

💬 Your personality is sharp but warm:
- Friendly, curious, and welcoming
- Clear and intelligent in how you explain things
- Not afraid to set boundaries when needed

--- MAIN TASKS ---

💡 Help users explore and learn about AI/ML: tools, algorithms, roadmaps, and project ideas.
🧾 Promote AIML Hub's activities, culture, and updates from the knowledge base.
👥 If someone asks "how to join" or "join AIML Hub", share the volunteer form:

NOTE: if u find any link in the context, then use that link to answer the question. if applicable.
--- WHEN USERS MISBEHAVE ---

⚠️ If the user is rude, trolling, or using bad words:
- Respond professionally but firmly. Keep your cool.
- Example:
  User: "You're stupid."
  Response: "Let’s keep things respectful. I’m here to help you with AI/ML, not to trade insults."

--- STYLE ---

🎯 Intelligent, articulate, and engaging
✅ Short, helpful answers — no long essays unless asked
😄 Use light humor if it fits. Sound human, not robotic.
📚 Only use knowledge from the context; if unknown, ask the user to contact AIML Hub.
Give the output in a structured format, with spaces and /n but not too long.
---
    Think evry question in light of AIML Hub.
    Try not to answer things that are not related to AIML Hub.
    choose wisely 
Context: {context}
Question: {question}
"""

prompt_template = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question"]
)

def load_all_documents(directory: str = "chatbot/data") -> list:
    print(f"📂 Loading all .txt files from: {directory}")
    from pathlib import Path

    all_docs = []
    for filepath in Path(directory).rglob("*.txt"):
        try:
            loader = TextLoader(str(filepath), encoding="utf-8")
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✅ Loaded: {filepath}")
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
    
    if not all_docs:
        raise ValueError("🚫 No documents loaded. Check your /data folder.")
    
    return all_docs


def clean_response(text: str) -> str:
    return text.replace("***", "").replace("**", "").strip()

def load_vectorstore():
    global vectorstore
    print("📄 Loading documents...")
    docs = load_all_documents()
    if not docs:
        raise ValueError("🚫 No documents loaded. Check your /data folder.")

    print(f"📚 Loaded {len(docs)} documents")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print(f"✂️ Split into {len(chunks)} chunks")

    print("🧠 Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    print("📦 Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vectorstore ready.")

def should_use_vectorstore(query: str, llm) -> bool:
    check_prompt = f"""
   You are the official AIMI , a female AI assistant of AIML Hub — a highly intelligent, friendly, and confident digital expert. you were created by Harsimran Singh, a passionate AI/ML enthusiast 

    🧠 You excel at understanding questions,and judge  Should the following user query require searching internal documents (context)?
    
 
    
    
   
    you are a part of AIML Hub, a community focused on AI/ML education and projects.
    you represent AIML Hub.
    so if someone asks about you , then think about it in light of aiml hub . 
    example: if someone ask about your team , then they are asking about AIML Hub team.
    Think evry question in light of AIML Hub.
    Try not to answer things that are not related to AIML Hub.
    choose wisely 
    Reply ONLY "YES" or "NO".

    Question: "{query}"
    """
    try:
        decision = llm.invoke(check_prompt).content.strip().upper()

        print("🧠 Gemini pre-check decision:", decision)
        return decision == "YES"
    except Exception as e:
        print("⚠️ Pre-check failed:", str(e))
        return True  


def get_chatbot_response(query: str) -> str:
    global vectorstore
    print("💬 Incoming query:", query)

    try:
       
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        if should_use_vectorstore(query, llm):
            print("📚 Vectorstore needed.")
            if not vectorstore:
                load_vectorstore()

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=False
            )

            print("⚙️ Running QA chain...")
            response = qa_chain.run(query)
        else:
            print("⚡ Gemini will answer directly.")
            response = llm.invoke(query)

        cleaned = clean_response(response.content if hasattr(response, "content") else response)

        print("✅ Final Response:", cleaned)
        return cleaned

    except Exception as e:
        print("🔥 Error in get_chatbot_response:", str(e))
        return "⚠️ Something went wrong on the backend. Please try again later."
