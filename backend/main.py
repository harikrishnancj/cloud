from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the current directory to sys.path to resolve imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rag_service import process_video, ask_question

app = FastAPI(title="YouTube RAG API")

# Configure CORS for frontend access
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://cloud-rho-nine.vercel.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/process-video")
async def process_video_endpoint(request: VideoRequest):
    try:
        result = process_video(request.url)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/ask")
async def ask_question_endpoint(request: QuestionRequest):
    try:
        answer = ask_question(request.question)
        return {"answer": answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
