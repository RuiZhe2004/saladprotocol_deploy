import google.generativeai as genai
from upstash_vector import Index
import os
import logging
from typing import List, Dict, Any
import hashlib
import json

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self):
        # Configure Gemini for embeddings
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Initialize Upstash Vector
        self.vector_index = Index(
            url=os.getenv("UPSTASH_VECTOR_REST_URL"),
            token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
        )
        
        # Initialize knowledge base if empty
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """
        Initialize the knowledge base with basic nutrition information.
        """
        try:
            # Check if knowledge base is empty
            stats = self.vector_index.info()
            
            if stats.dimension == 0:  # Empty database
                logger.info("Initializing knowledge base with nutrition information...")
                self._add_initial_knowledge()
                
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
    
    def _add_initial_knowledge(self):
        """
        Add initial nutrition knowledge to the vector database.
        """
        initial_knowledge = [
            {
                "content": "Protein is essential for muscle building and repair. Adults should consume 0.8-1.2g per kg of body weight daily. Good sources include lean meats, fish, eggs, legumes, and dairy products.",
                "category": "macronutrients"
            },
            {
                "content": "Carbohydrates are the body's primary energy source. Complex carbs from whole grains, fruits, and vegetables provide sustained energy and fiber. Simple carbs should be limited.",
                "category": "macronutrients"
            },
            {
                "content": "Healthy fats are crucial for hormone production and nutrient absorption. Include omega-3 fatty acids from fish, nuts, and seeds. Limit saturated and trans fats.",
                "category": "macronutrients"
            },
            {
                "content": "Vitamin D is essential for bone health and immune function. Sources include sunlight exposure, fatty fish, and fortified foods. Many people are deficient.",
                "category": "vitamins"
            },
            {
                "content": "Fiber aids digestion and helps maintain healthy blood sugar levels. Adults need 25-35g daily from fruits, vegetables, whole grains, and legumes.",
                "category": "nutrients"
            },
            {
                "content": "Hydration is crucial for all bodily functions. Aim for 8-10 glasses of water daily, more if active. Water needs vary based on activity level and climate.",
                "category": "hydration"
            },
            {
                "content": "Meal timing can affect metabolism and energy levels. Eating regular meals helps maintain stable blood sugar. Consider eating every 3-4 hours.",
                "category": "meal_planning"
            },
            {
                "content": "Portion control is key for weight management. Use the plate method: half vegetables, quarter protein, quarter whole grains.",
                "category": "portion_control"
            },
            {
                "content": "Antioxidants protect cells from damage. Colorful fruits and vegetables are rich in antioxidants. Aim for a variety of colors in your diet.",
                "category": "nutrients"
            },
            {
                "content": "Calcium is essential for bone health. Good sources include dairy products, leafy greens, and fortified plant-based milks. Vitamin D enhances calcium absorption.",
                "category": "minerals"
            }
        ]
        
        for knowledge in initial_knowledge:
            try:
                self.add_knowledge(knowledge["content"], knowledge["category"])
            except Exception as e:
                logger.error(f"Error adding initial knowledge: {str(e)}")
    
    async def add_knowledge(self, content: str, category: str = "general") -> str:
        """
        Add knowledge to the vector database.
        """
        try:
            # Generate embedding using Gemini
            embedding_result = genai.embed_content(
                model="models/embedding-001",
                content=content,
                task_type="retrieval_document"
            )
            
            # Create unique ID for the knowledge
            knowledge_id = hashlib.md5(content.encode()).hexdigest()
            
            # Prepare metadata
            metadata = {
                "content": content,
                "category": category,
                "created_at": datetime.now().isoformat()
            }
            
            # Upsert to vector database
            self.vector_index.upsert(
                vectors=[{
                    "id": knowledge_id,
                    "values": embedding_result['embedding'],
                    "metadata": metadata
                }]
            )
            
            logger.info(f"Added knowledge to vector DB: {knowledge_id}")
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
            raise e
    
    async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant knowledge using semantic similarity.
        """
        try:
            # Generate embedding for the query
            embedding_result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            
            # Search vector database
            search_results = self.vector_index.query(
                vector=embedding_result['embedding'],
                top_k=limit,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in search_results.matches:
                formatted_results.append({
                    "content": match.metadata.get("content", ""),
                    "category": match.metadata.get("category", "general"),
                    "score": match.score,
                    "id": match.id
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {str(e)}")
            return []
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the vector database.
        """
        try:
            self.vector_index.delete(ids=[knowledge_id])
            logger.info(f"Deleted knowledge from vector DB: {knowledge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge: {str(e)}")
            return False
