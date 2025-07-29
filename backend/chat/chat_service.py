import datetime
import google.generativeai as genai
from typing import Dict, Any, Optional, List
import os
import logging
import json
import joblib  # For saving and loading the model
import numpy as np
import pandas as pd  # For DataFrame creation
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction
from sklearn.pipeline import Pipeline #Import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)
DT_Model = DT_Model = "chat/inference_engine_model.joblib"

# Define RULE_TYPES here, or import it if defined elsewhere.
RULE_TYPES = [  #  Copied from train_inference_engine.py
    "weight_loss", "muscle_gain", "general_health", "specific_nutrient_recommendation",
    "ask_for_more_information", "hydration_recommendation", "sleep_improvement",
    "stress_management_nutrition", "energy_boosting_foods", "skin_health_nutrition",
    "digestive_health", "immune_boosting", "bone_health", "eye_health_nutrition",
    "brain_health", "hair_health_nutrition", "pregnancy", "diabetes",
    "high_cholesterol", "low_blood_pressure", "kidney_issues", "anxiety",
]


class ChatService:
    def __init__(self, firebase_service, vector_service):
        self.firebase_service = firebase_service
        self.vector_service = vector_service

        # Load the Pipeline model
        self.model_components = self._load_model_components(DT_Model)
        self.pipeline = self.model_components.get('pipeline') # Use .get to avoid KeyError
        self.rule_types = self.model_components.get('rule_types', RULE_TYPES)  # Access RULE_TYPES from loaded components, default to RULE_TYPES if not found

        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Renamed to avoid confusion

    def _load_model_components(self, model_path: str) -> Dict[str, Any]:
        """Loads the complete model pipeline from a file."""
        try:
            model_components = joblib.load(model_path)
            logger.info(f"Model components successfully loaded from {model_path}")

            return model_components

        except FileNotFoundError:
            logger.warning(f"Model components not found at {model_path}. Creating default components.")
            # Handle the file not found case
            return self._create_default_model()  # Call helper method

        except Exception as e:
            logger.error(f"Error loading model components: {e}. Creating default components.  Error: {e}")
            #Handle any other exception like corrupted files etc.
            return self._create_default_model()

    def _create_default_model(self) -> Dict[str, Any]:
        """Creates a default model pipeline if loading fails."""
        try:
            tfidf = TfidfVectorizer(max_features=50) #Set to same number as training
            scaler = StandardScaler()

            # Ensure these feature names match what's expected in _prepare_model_features
            numerical_features = ['user_age', 'bmi']  #Numerical features
            text_feature = 'message'

            #Dummy ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', scaler, numerical_features),  # Apply scaler to numerical features
                    ('tfidf', tfidf, text_feature) #Apply tfidf to the message
                ])

            classifier = DecisionTreeClassifier()
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])

            # Fit the pipeline on some dummy data to prevent "NotFittedError"
            # Ensure the dummy data matches the expected input features
            dummy_data = pd.DataFrame({
                'message': ["test message"],
                'user_age': [30],
                'bmi': [22.5]
            })
            #Ensure right order of columns for the dataframe
            dummy_data = dummy_data[['message', 'user_age', 'bmi']]

            # Dummy target data
            dummy_target = [0]  # Example: the first class

            try:
                pipeline.fit(dummy_data, dummy_target) #Pass target data now
                logger.info("Successfully fitted the default pipeline.")
            except Exception as fit_err:
                logger.error(f"Failed to fit the default pipeline: {fit_err}")
                pipeline = None  # Set pipeline to None if fitting fails

            return {
                'pipeline': pipeline,
                'rule_types': RULE_TYPES #Set a default rul_type
            }

        except Exception as e:
            logger.error(f"Error creating default model: {e}")
            return {  # Return a minimal, non-functional model to avoid cascading errors
                'pipeline': None,
                'rule_types': RULE_TYPES
            }

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

            # Prepare features for the  model
            features = self._prepare_model_features(message, user_profile)

            # Make prediction with the Model
            model_prediction = self._make_model_prediction(features)

            # Build context-aware prompt
            prompt = await self._build_prompt(
                message=message,
                user_profile=user_profile,
                relevant_knowledge=relevant_knowledge,
                last_food_analysis=last_food_analysis,
                conversation_history=conversation_history,
                model_prediction=model_prediction
            )

            # Generate response using Gemini
            response = self.gemini_model.generate_content(prompt)  # Use the renamed variable

            # Store conversation in Firebase for future reference
            await self._store_conversation(username, message, response.text)

            return response.text

        except Exception as e:
            logger.error(f"Chat service error: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."

    def _prepare_model_features(
            self,
            message: str,
            user_profile: Dict[str, Any],
    ) -> pd.DataFrame:
        """Prepares features for the  model."""
        # Create a dataframe from the available information
        user_age = user_profile.get("age", 0)
        user_height = user_profile.get("height", 0)
        user_weight = user_profile.get("weight", 0)
        bmi = user_weight / ((user_height/100)**2) if user_height > 0 else 0.0

        # Ensure the DataFrame includes ALL features used during training
        data = {'message': [message], 'user_age': [user_age], 'bmi': [bmi]}
        df = pd.DataFrame(data)

        # Ensure right order of columns for the dataframe and matching training
        df = df[['message', 'user_age', 'bmi']]

        #Print data types BEFORE the next if statement
        print("Data types in _prepare_model_features:\n", df.dtypes)

        #Check for NaN
        if df.isnull().any().any():
            print("WARNING: NaN values detected in features DataFrame before prediction! Filling with 0.")
            df = df.fillna(0)  # Or handle NaNs appropriately

        return df

    def _make_model_prediction(self, features: pd.DataFrame) -> str:
        """Makes a prediction using the  model."""
        try:
             # Check if the pipeline is None
            if self.pipeline is None:
                logger.warning("Pipeline is None, returning default 'general_health'.")
                return "general_health"

            prediction_encoded = self.pipeline.predict(features)[0]  # Get encoded prediction
            prediction = self.rule_types[prediction_encoded] #Decode the type

            return prediction  # Return the decoded prediction

        except Exception as e:
            logger.error(f"Error making  model prediction: {e}")
            return "general_health"  # Return a default action or advice

    async def _build_prompt(
            self,
            message: str,
            user_profile: Dict[str, Any],
            relevant_knowledge: List[Dict[str, Any]],
            last_food_analysis: Optional[Dict[str, Any]] = None,
            conversation_history: Optional[List[Dict[str, Any]]] = None,
            model_prediction: Optional[str] = None  # Add decision tree output to the arguments
    ) -> str:
        """
        Build a comprehensive prompt with user context and knowledge base information.
        """
        prompt_parts = []

        # System prompt
        prompt_parts.append("""
You are Salad Protocol, an expert AI nutritionist. You provide personalized, evidence-based nutrition advice.

Key guidelines:
- RESPOND IN PLAIN TEXT ONLY, NO MARKDOWN OR FORMATTING DONT INCLUDE ANY * OR / OR BOLDING OR ITALICS
- Always be supportive and encouraging
- Provide specific, actionable advice
- Use the user's profile data to personalize responses
- Reference relevant knowledge from your knowledge base
- If discussing food analysis results, be specific about the nutritional content
- Encourage healthy eating habits and lifestyle choices
- If you don't have enough information, ask clarifying questions
- Respond in plain text, without any markdown formatting like bolding or italics.
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

        # Model prediction context
        if model_prediction:
            prompt_parts.append(f"Based on model analysis, the recommended advice category is:\n{model_prediction}")  # Modified Prompt

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
                "timestamp": datetime.datetime.now().isoformat()
            }

            await self.firebase_service.store_conversation(conversation_data)

        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")