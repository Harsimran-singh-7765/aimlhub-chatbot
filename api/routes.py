 # api/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from chatbot.chatbot import get_chatbot_response
from fastapi.middleware.cors import CORSMiddleware
app = APIRouter()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    reply = get_chatbot_response(request.message)
    return {"response": reply}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}
