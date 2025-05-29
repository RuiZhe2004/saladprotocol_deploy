import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, Any, Optional
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class FirebaseService:
    def __init__(self):
        # Initialize Firebase Admin SDK
        if not firebase_admin._apps:
            # Use service account key file or default credentials
            cred = credentials.Certificate(r"firebase_credentials.json") #credentials.ApplicationDefault()  
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
    
    async def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user data from Firestore.
        """
        try:
            doc_ref = self.db.collection('users').document(username)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting user {username}: {str(e)}")
            raise e
    
    async def create_user(self, username: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new user in Firestore.
        """
        try:
            doc_ref = self.db.collection('users').document(username)
            doc_ref.set(user_data)
            return user_data
            
        except Exception as e:
            logger.error(f"Error creating user {username}: {str(e)}")
            raise e
    
    async def update_user(self, username: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user data in Firestore.
        """
        try:
            doc_ref = self.db.collection('users').document(username)
            doc_ref.update(update_data)
            
            # Return updated user data
            updated_doc = doc_ref.get()
            return updated_doc.to_dict()
            
        except Exception as e:
            logger.error(f"Error updating user {username}: {str(e)}")
            raise e
    
    async def store_food_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """
        Store food analysis result in Firestore.
        """
        try:
            doc_ref = self.db.collection('food_analyses').add(analysis_data)
            return doc_ref[1].id
            
        except Exception as e:
            logger.error(f"Error storing food analysis: {str(e)}")
            raise e
    
    async def store_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """
        Store conversation in Firestore.
        """
        try:
            doc_ref = self.db.collection('conversations').add(conversation_data)
            return doc_ref[1].id
            
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            raise e
    
    async def get_user_food_history(self, username: str, limit: int = 10) -> list:
        """
        Get user's food analysis history.
        """
        try:
            query = (self.db.collection('food_analyses')
                    .where('username', '==', username)
                    .order_by('timestamp', direction=firestore.Query.DESCENDING)
                    .limit(limit))
            
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
            
        except Exception as e:
            logger.error(f"Error getting food history for user {username}: {str(e)}")
            raise e
