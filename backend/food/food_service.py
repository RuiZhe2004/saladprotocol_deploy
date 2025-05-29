import aiohttp
import json
from typing import Dict, Any
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class FoodService:
    def __init__(self, firebase_service):
        self.firebase_service = firebase_service
        # URL for your custom food recognition model
        self.custom_model_url = os.getenv("CUSTOM_MODEL_URL", "http://localhost:8001")
    
    async def analyze_food_image(self, image_data: bytes, filename: str, username: str) -> Dict[str, Any]:
        """
        Analyze food image using custom model and store results.
        """
        try:
            # Upload image to Firebase Storage
            image_url = await self.firebase_service.upload_image(
                image_data=image_data,
                filename=f"food_images/{username}/{datetime.now().isoformat()}_{filename}",
                username=username
            )
            
            # Call custom food recognition model
            analysis_result = await self._call_custom_model(image_data)
            
            # Store analysis result in Firebase
            analysis_data = {
                "username": username,
                "image_url": image_url,
                "analysis_result": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.firebase_service.store_food_analysis(analysis_data)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Food analysis service error: {str(e)}")
            raise e
    
    async def _call_custom_model(self, image_data: bytes) -> Dict[str, Any]:
        """
        Call your custom food recognition model API.
        """
        try:
            # Prepare the request to your custom model
            async with aiohttp.ClientSession() as session:
                # Create form data for the image
                data = aiohttp.FormData()
                data.add_field('image', image_data, content_type='image/jpeg')
                
                # Call your custom model endpoint
                async with session.post(f"{self.custom_model_url}/analyze", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Custom model error: {error_text}")
                        
        except Exception as e:
            logger.error(f"Custom model call error: {str(e)}")
            # Return mock data for development/testing
            return self._get_mock_analysis()
    
    def _get_mock_analysis(self) -> Dict[str, Any]:
        """
        Mock food analysis for development/testing purposes.
        Replace this with actual model integration.
        """
        return {
            "food_items": [
                {
                    "name": "Grilled Chicken Breast",
                    "calories": 165,
                    "protein": 31.0,
                    "carbs": 0.0,
                    "fat": 3.6,
                    "portion_size": "100g"
                },
                {
                    "name": "Mixed Vegetables",
                    "calories": 45,
                    "protein": 2.0,
                    "carbs": 9.0,
                    "fat": 0.3,
                    "portion_size": "150g"
                }
            ],
            "total_calories": 210,
            "total_protein": 33.0,
            "total_carbs": 9.0,
            "total_fat": 3.9,
            "confidence_score": 0.85
        }
