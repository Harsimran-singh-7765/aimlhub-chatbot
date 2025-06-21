from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str


def get_chatbot_response(message: str) -> str:
    return f"You said: {message}"

@app.post("/chat")
async def chat(request: ChatRequest):
    reply = get_chatbot_response(request.message)
    return {"response": reply}

@app.get("/health")
def health_check():
    return {"status": "ok"}
