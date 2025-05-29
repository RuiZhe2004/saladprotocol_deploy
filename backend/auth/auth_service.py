from typing import Dict, Any, Optional
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self, firebase_service):
        self.firebase_service = firebase_service
    
    async def login(self, username: str) -> Dict[str, Any]:
        """
        Handle user login. Check if user exists, create if new.
        """
        try:
            # Check if user exists in Firebase
            user_data = await self.firebase_service.get_user(username)
            
            if user_data:
                # Existing user
                return {
                    "is_new_user": False,
                    "user": user_data
                }
            else:
                # New user - create basic record
                new_user = {
                    "username": username,
                    "created_at": datetime.now().isoformat(),
                    "profile_completed": False
                }
                
                await self.firebase_service.create_user(username, new_user)
                
                return {
                    "is_new_user": True,
                    "user": new_user
                }
                
        except Exception as e:
            logger.error(f"Login service error: {str(e)}")
            raise e
    
    async def setup_profile(self, username: str, birthday: str, height: float, weight: float) -> Dict[str, Any]:
        """
        Complete user profile setup.
        """
        try:
            # Calculate age from birthday
            birth_date = datetime.strptime(birthday, "%Y-%m-%d").date()
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            # Update user profile
            profile_data = {
                "birthday": birthday,
                "height": height,
                "weight": weight,
                "age": age,
                "profile_completed": True,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_user = await self.firebase_service.update_user(username, profile_data)
            
            return updated_user
            
        except Exception as e:
            logger.error(f"Profile setup service error: {str(e)}")
            raise e
