from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import json
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Food Recognition Model API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = load_model("C:/Users\Asus TUF/SaladProtocol_v2/backend/custom_model/food_detection.h5")

# Load class labels
with open("C:/Users/Asus TUF/SaladProtocol_v2/backend/custom_model/class_labels.json", "r") as f:
    class_labels = json.load(f)
class_labels = {int(k): v for k, v in class_labels.items()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "food_recognition_v1"}

@app.post("/analyze")
async def analyze_food_image(image: UploadFile = File(...)):
    """
    Analyze food image and return predicted class and confidence.
    """
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        processed_image = preprocess_image(pil_image)
        predictions = model.predict(processed_image)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(predictions[0]))

        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Food analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
async def predict_image(url: str = Query(..., description="Image URL to classify")):
    """
    Predict food class from an image URL.
    """
    try:
        img = preprocess_image_from_url(url)
        predictions = model.predict(img)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(predictions[0]))

        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess PIL image for model inference.
    """
    image = image.resize((224, 224))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def preprocess_image_from_url(url: str) -> np.ndarray:
    """
    Preprocess image from URL for model inference.
    """
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
