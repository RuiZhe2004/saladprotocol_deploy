import google.generativeai as genai
from typing import Dict, Any, Optional, List
import os
import logging
import json

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, firebase_service, vector_service):
        self.firebase_service = firebase_service
        self.vector_service = vector_service
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def get_response(
        self, 
        message: str, 
        username: str, 
        last_food_analysis: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate AI response using Gemini with RAG and user context.
        """
        try:
            # Get user profile for personalization
            user_profile = await self.firebase_service.get_user(username)
            
            # Search knowledge base for relevant information
            relevant_knowledge = await self.vector_service.search_knowledge(message, limit=3)
            
            # Build context-aware prompt
            prompt = await self._build_prompt(
                message=message,
                user_profile=user_profile,
                relevant_knowledge=relevant_knowledge,
                last_food_analysis=last_food_analysis,
                conversation_history=conversation_history
            )
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Store conversation in Firebase for future reference
            await self._store_conversation(username, message, response.text)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Chat service error: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
    
    async def _build_prompt(
        self,
        message: str,
        user_profile: Dict[str, Any],
        relevant_knowledge: List[Dict[str, Any]],
        last_food_analysis: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build a comprehensive prompt with user context and knowledge base information.
        """
        prompt_parts = []
        
        # System prompt
        prompt_parts.append("""
You are Salad Protocol, an expert AI nutritionist. You provide personalized, evidence-based nutrition advice.

Key guidelines:
- Always be supportive and encouraging
- Provide specific, actionable advice
- Use the user's profile data to personalize responses
- Reference relevant knowledge from your knowledge base
- If discussing food analysis results, be specific about the nutritional content
- Encourage healthy eating habits and lifestyle choices
- If you don't have enough information, ask clarifying questions
        """)
        
        # User profile context
        if user_profile:
            profile_info = f"""
User Profile:
- Username: {user_profile.get('username', 'Unknown')}
- Age: {user_profile.get('age', 'Unknown')} years
- Height: {user_profile.get('height', 'Unknown')} cm
- Weight: {user_profile.get('weight', 'Unknown')} kg
            """
            prompt_parts.append(profile_info)
        
        # Knowledge base context
        if relevant_knowledge:
            knowledge_context = "Relevant nutrition knowledge:\n"
            for i, knowledge in enumerate(relevant_knowledge, 1):
                knowledge_context += f"{i}. {knowledge.get('content', '')}\n"
            prompt_parts.append(knowledge_context)
        
        # Food analysis context
        if last_food_analysis:
            food_context = f"""
Recent food analysis:
{json.dumps(last_food_analysis, indent=2)}

The user may be asking questions about this food analysis.
            """
            prompt_parts.append(food_context)
        
        # Conversation history
        if conversation_history:
            history_context = "Recent conversation:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = "User" if msg.get('role') == 'user' else "Assistant"
                history_context += f"{role}: {msg.get('content', '')}\n"
            prompt_parts.append(history_context)
        
        # Current user message
        prompt_parts.append(f"\nUser's current message: {message}")
        prompt_parts.append("\nProvide a helpful, personalized response:")
        
        return "\n".join(prompt_parts)
    
    async def _store_conversation(self, username: str, user_message: str, ai_response: str):
        """
        Store conversation in Firebase for future reference.
        """
        try:
            conversation_data = {
                "username": username,
                "user_message": user_message,
                "ai_response": ai_response,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.firebase_service.store_conversation(conversation_data)
            
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            # Don't raise error as this is not critical for user experience
