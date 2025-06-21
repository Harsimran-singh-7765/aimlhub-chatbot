 # api/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from chatbot.chatbot import get_chatbot_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    reply = get_chatbot_response(request.message)
    return {"response": reply}

