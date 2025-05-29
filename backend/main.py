from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from datetime import datetime, date
import json
import logging

# Import custom modules (to be created)
from auth.auth_service import AuthService
from chat.chat_service import ChatService
from food.food_service import FoodService
from database.firebase_service import FirebaseService
from vector_db.vector_service import VectorService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Salad Protocol Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
firebase_service = FirebaseService()
vector_service = VectorService()
auth_service = AuthService(firebase_service)
chat_service = ChatService(firebase_service, vector_service)
food_service = FoodService(firebase_service)

# Pydantic models
class LoginRequest(BaseModel):
    username: str

class ProfileSetupRequest(BaseModel):
    username: str
    birthday: str
    height: float
    weight: float

class ChatRequest(BaseModel):
    message: str
    username: str
    last_food_analysis: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None

class LoginResponse(BaseModel):
    is_new_user: bool
    user: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    try:
        result = await auth_service.login(request.username)
        return LoginResponse(**result)
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/setup-profile")
async def setup_profile(request: ProfileSetupRequest):
    try:
        user = await auth_service.setup_profile(
            username=request.username,
            birthday=request.birthday,
            height=request.height,
            weight=request.weight
        )
        return {"user": user}
    except Exception as e:
        logger.error(f"Profile setup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = await chat_service.get_response(
            message=request.message,
            username=request.username,
            last_food_analysis=request.last_food_analysis,
            conversation_history=request.conversation_history
        )
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Food analysis endpoint
@app.post("/food/analyze")
async def analyze_food(
    image: UploadFile = File(...),
    username: str = Form(...)
):
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Analyze food using custom model
        analysis_result = await food_service.analyze_food_image(
            image_data=image_data,
            filename=image.filename,
            username=username
        )
        
        return analysis_result
    except Exception as e:
        logger.error(f"Food analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge base management endpoints
@app.post("/knowledge/add")
async def add_knowledge(content: str, category: str = "general"):
    try:
        result = await vector_service.add_knowledge(content, category)
        return {"success": True, "id": result}
    except Exception as e:
        logger.error(f"Add knowledge error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/search")
async def search_knowledge(query: str, limit: int = 5):
    try:
        results = await vector_service.search_knowledge(query, limit)
        return {"results": results}
    except Exception as e:
        logger.error(f"Search knowledge error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
