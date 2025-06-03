# Salad Protocol - AI Nutritionist Chatbot

A comprehensive AI-powered nutritionist chatbot with food image recognition capabilities, built with Next.js frontend and Python FastAPI backend.

## Features

- 🤖 **AI Nutritionist Chat**: Personalized nutrition advice using Gemini API
- 📸 **Food Image Recognition**: Custom ML model for food identification and calorie estimation
- 🔍 **Knowledge Base**: RAG system with Upstash Vector for semantic search
- 👤 **User Profiles**: Personalized responses based on user data (age, height, weight)
- 🔥 **Firebase Integration**: Secure data storage and image hosting
- 🥗 **Healthy Design**: Green-themed UI promoting wellness

## Tech Stack

### Frontend
- Next.js 14 with App Router
- React with TypeScript
- Tailwind CSS + shadcn/ui
- Responsive design

### Backend
- Python FastAPI
- Gemini API for chat and embeddings
- Upstash Vector for knowledge base
- Firebase for data storage
- Custom food recognition model

## Project Structure

\`\`\`
salad-protocol/
├── frontend/                 # Next.js frontend
│   ├── app/
│   │   ├── page.tsx         # Login page
│   │   ├── profile-setup/   # Profile setup
│   │   ├── chat/           # Main chat interface
│   │   └── api/            # API routes
│   └── components/         # UI components
├── backend/                 # Python FastAPI backend
│   ├── main.py             # Main FastAPI app
│   ├── auth/               # Authentication services
│   ├── chat/               # Chat services
│   ├── food/               # Food analysis services
│   ├── database/           # Firebase integration
│   ├── vector_db/          # Vector database services
│   └── custom_model/       # Food recognition model
└── README.md
\`\`\`

## Setup Instructions

### Prerequisites

1. **API Keys & Services**:
   - Gemini API key
   - Firebase project with Firestore and Storage
   - Upstash Vector database
   - Service account key for Firebase Admin

### Backend Setup

1. **Navigate to backend directory**:
   \`\`\`
   cd backend
   \`\`\`

2. **Create virtual environment**:
   \`\`\`
   python -m venv venv (No need if you've created already)
   venv\Scripts\activate 
   \`\`\`

3. **Install dependencies**:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

4. **Set up Firebase**:
   - Download your Firebase service account key
   - Place it in the backend directory as `serviceAccountKey.json`
   - Update `GOOGLE_APPLICATION_CREDENTIALS` in `.env`

5. **Run the backend**:
   \`\`\`
   uvicorn main:app --reload --port 8000
   \`\`\`

6. **Run the custom model API** (in a separate terminal):
   \`\`\`
   cd custom_model
   python food_recognition_api.py
   \`\`\`

### Frontend Setup

1. **Navigate to frontend directory**:
   \`\`\`
   cd frontend
   \`\`\`

2. **Install dependencies**:
   \`\`\`
   npm install
   \`\`\`


3. **Run the frontend**:
   \`\`\`bash
   npm run dev
   \`\`\`


### Backend (.env)
\`\`\`env
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=./serviceAccountKey.json
FIREBASE_STORAGE_BUCKET=your_project.appspot.com
UPSTASH_VECTOR_REST_URL=your_upstash_url
UPSTASH_VECTOR_REST_TOKEN=your_upstash_token
CUSTOM_MODEL_URL=http://localhost:8001
\`\`\`

### Frontend (.env.local)
\`\`\`env
PYTHON_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_FIREBASE_API_KEY=your_firebase_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
\`\`\`

## Custom Food Recognition Model Integration

The system is designed to integrate with your custom food recognition model:

1. **Model API**: Implement your model in `backend/custom_model/food_recognition_api.py`
2. **Expected Input**: Image file (JPEG/PNG)
3. **Expected Output**: JSON with nutritional breakdown:
   \`\`\`json
   {
     "food_items": [
       {
         "name": "Food Name",
         "calories": 100,
         "protein": 10.0,
         "carbs": 15.0,
         "fat": 5.0,
         "portion_size": "100g"
       }
     ],
     "total_calories": 100,
     "total_protein": 10.0,
     "total_carbs": 15.0,
     "total_fat": 5.0
   }
   \`\`\`

## Usage

1. **Login**: Enter username (creates account if new)
2. **Profile Setup**: New users provide age, height, weight
3. **Chat**: Ask nutrition questions, get personalized advice
4. **Food Analysis**: Upload food photos for nutritional analysis
5. **Integrated Experience**: Ask questions about analyzed food

## Key Features Implementation

### RAG System
- Knowledge base stored in Upstash Vector
- Semantic search using Gemini embeddings
- Context-aware responses

### Personalization
- User profile integration in chat prompts
- Age, height, weight-based recommendations
- Food history tracking

### Food Recognition
- Custom model integration ready
- Firebase image storage
- Nutritional data extraction

## Development Notes

- The custom food recognition model currently returns mock data
- Replace the mock implementation with your trained model
- Knowledge base can be expanded by adding more nutrition content
- All user data is securely stored in Firebase

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
