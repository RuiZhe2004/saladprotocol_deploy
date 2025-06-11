from google.cloud import storage
import aiohttp
import json
from typing import Dict, Any
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class FoodService:
    def __init__(self, firebase_service, vector_service):
        self.firebase_service = firebase_service
        self.vector_service = vector_service
        self.custom_model_url = os.getenv("CUSTOM_MODEL_URL", "http://localhost:8001")
        self.gcs_bucket_name = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")
    
    async def analyze_food_image(self, image_data: bytes, filename: str, username: str) -> Dict[str, Any]:
        """
        Analyze food image using custom model and return results in frontend format.
        """
        try:
            # Step 1: Upload image to Google Cloud Storage
            image_url = self._upload_to_gcs(image_data, filename, username)
            
            # Step 2: Call custom food recognition model
            model_result = await self._call_custom_model(image_data)
            food_name = model_result.get("predicted_class", "unknown")
            confidence = model_result.get("confidence", 0.0)

            # Step 3: Get nutrition info
            nutrition_info = await self._get_nutrition_info(food_name)

            # Step 4: Integrate with knowledge base (optional, not used in frontend display)
            try:
                knowledge_context = await self._generate_knowledge_context({
                    "food_items": [{"name": food_name}]
                })
            except Exception as ke:
                logger.error(f"Knowledge context error (non-critical): {str(ke)}")
                knowledge_context = []

            # Step 5: Return in frontend format
            analysis_result = {
                "image_url": image_url,
                "food_items": [
                    {
                        "name": food_name,
                        "calories": nutrition_info.get("calories", 0),
                        "protein": nutrition_info.get("protein", 0),
                        "carbs": nutrition_info.get("carbs", 0),
                        "fat": nutrition_info.get("fat", 0),
                        "portion_size": nutrition_info.get("portion_size", "100g"),
                        "confidence": confidence
                    }
                ],
                "total_calories": nutrition_info.get("calories", 0),
                "total_protein": nutrition_info.get("protein", 0),
                "total_carbs": nutrition_info.get("carbs", 0),
                "total_fat": nutrition_info.get("fat", 0),
                "confidence_score": confidence
            }
            return analysis_result

        except Exception as e:
            logger.error(f"Food analysis service error: {str(e)}")
            raise e

    async def _call_custom_model_with_url(self, image_url: str) -> Dict[str, Any]:
        """
        Call your custom food recognition model API using the image URL.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Use the /predict endpoint with the image URL
                url = f"{self.custom_model_url}/predict?url={image_url}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        model_prediction = await response.json()
                        
                        # Transform the response to match frontend expectations
                        food_name = model_prediction.get("predicted_class", "unknown")
                        confidence = model_prediction.get("confidence", 0.0)
                        
                        # Get nutrition data for this food
                        nutrition_info = await self._get_nutrition_info(food_name)
                        
                        # Return in the format expected by frontend
                        return {
                            "food_items": [
                                {
                                    "name": food_name,
                                    "calories": nutrition_info.get("calories", 0),
                                    "protein": nutrition_info.get("protein", 0),
                                    "carbs": nutrition_info.get("carbs", 0),
                                    "fat": nutrition_info.get("fat", 0),
                                    "portion_size": nutrition_info.get("portion_size", "100g"),
                                    "confidence": confidence
                                }
                            ],
                            "total_calories": nutrition_info.get("calories", 0),
                            "total_protein": nutrition_info.get("protein", 0),
                            "total_carbs": nutrition_info.get("carbs", 0),
                            "total_fat": nutrition_info.get("fat", 0)
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Custom model error: {error_text}")
                        
        except Exception as e:
            logger.error(f"Custom model call error: {str(e)}")
            return self._get_mock_analysis()

    async def _get_nutrition_info(self, food_name: str) -> Dict[str, Any]:
        """
        Get nutrition information for the predicted food using USDA API.
        """
        api_key = os.getenv("USDA_API_KEY")
        if not api_key:
            logger.warning("USDA API key not set. Returning default values.")
            return {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "portion_size": "unknown"}

        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"USDA API error: {response.status}")
                        return {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "portion_size": "unknown"}

                    data = await response.json()
                    for item in data.get('foods', []):
                        fdc_id = item.get('fdcId')
                        nutrients = item.get('foodNutrients', [])
                        logger.info(f"FDC ID: {fdc_id}")
                        
                    # logger.info(f"Querying USDA API for: {food_name}")
                    # logger.info(f"USDA API URL: {url}")
                    # logger.info(f"USDA response data: {data}")
                    foods = data.get("foods", [])
                    if not foods:
                        return {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "portion_size": "unknown"}
                    
                    # food_nutrients = foods[0].get("foodNutrients", [])

                    for food in foods:
                        nutrients = food.get("foodNutrients", [])
                        if any(n.get("value", 0) > 0 for n in nutrients):
                            food_nutrients = nutrients
                            break
                    else:
                        return {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "portion_size": "unknown"}

                    calories = protein = carbs = fat = 0

                    for nutrient in food_nutrients:
                        name = nutrient.get("nutrientName", "").lower()
                        value = nutrient.get("value", 0)
                        if "energy" in name or "kcal" in name:
                            calories = value
                        elif "protein" in name:
                            protein = value
                        elif "carbohydrate" in name:
                            carbs = value
                        elif "total lipid (fat)" in name:
                            fat = value

                    return {
                        "calories": calories,
                        "protein": protein,
                        "carbs": carbs,
                        "fat": fat,
                        "portion_size": "100g"
                    }



                    # nutrients = {n["nutrientName"].lower(): n["value"] for n in foods[0].get("foodNutrients", [])}

                    # return {
                    #     "calories": nutrients.get("energy", 0),
                    #     "protein": nutrients.get("protein", 0),
                    #     "carbs": nutrients.get("carbohydrate, by difference", 0),
                    #     "fat": nutrients.get("total lipid (fat)", 0),
                    #     "portion_size": "100g"
                    # }
        except Exception as e:
            logger.error(f"Nutrition API call failed: {str(e)}")
            return {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "portion_size": "unknown"}

    # async def _get_nutrition_info(self, food_name: str) -> Dict[str, Any]:
    #     """
    #     Get nutrition information for the predicted food.
    #     """
    #     # This is a simplified version - you could integrate with a nutrition API
    #     nutrition_data = {
    #         "apple": {"calories": 52, "protein": 0.3, "carbs": 14.0, "fat": 0.2, "portion_size": "100g"},
    #         "banana": {"calories": 89, "protein": 1.1, "carbs": 22.8, "fat": 0.3, "portion_size": "100g"},
    #         "carrot": {"calories": 41, "protein": 0.9, "carbs": 9.6, "fat": 0.2, "portion_size": "100g"},
    #         "orange": {"calories": 47, "protein": 0.9, "carbs": 11.8, "fat": 0.1, "portion_size": "100g"},
    #         "tomato": {"calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2, "portion_size": "100g"},
    #         # Add more foods or call an actual API like USDA FoodData Central
    #     }
        
    #     return nutrition_data.get(food_name.lower(), 
    #                             {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "portion_size": "unknown"})
    
    def _upload_to_gcs(self, image_data: bytes, filename: str, username: str) -> str:
        """
        Upload image to Google Cloud Storage and return the public URL.
        """
        try:
            # Initialize Google Cloud Storage client
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcs_bucket_name)
            
            # Create a unique file path
            blob_path = f"food_images/{username}/{datetime.now().isoformat()}_{filename}"
            blob = bucket.blob(blob_path)
            
            # Upload the image
            blob.upload_from_string(image_data, content_type="image/jpeg")
            
            # Make the file publicly accessible
            blob.make_public()
            
            logger.info(f"Image uploaded to GCS: {blob.public_url}")
            return blob.public_url
        except Exception as e:
            logger.error(f"Error uploading image to GCS: {str(e)}")
            raise e
    
    async def _call_custom_model(self, image_data: bytes) -> Dict[str, Any]:
        """
        Call your custom food recognition model API.
        """
        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('image', image_data, content_type='image/jpeg')
                
                async with session.post(f"{self.custom_model_url}/analyze", data=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Custom model error: {error_text}")
                        
        except Exception as e:
            logger.error(f"Custom model call error: {str(e)}")
            return self._get_mock_analysis()
    
    async def _generate_knowledge_context(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate knowledge context based on food analysis results.
        """
        try:
            food_items = [item["name"] for item in analysis_result.get("food_items", [])]
            query = f"Nutrition information for {', '.join(food_items)}"
            knowledge_results = await self.vector_service.search_knowledge(query, limit=5)
            return knowledge_results
        except Exception as e:
            logger.error(f"Knowledge context generation error: {str(e)}")
            return []

    def _get_mock_analysis(self) -> Dict[str, Any]:
        """
        Mock food analysis for development/testing purposes.
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