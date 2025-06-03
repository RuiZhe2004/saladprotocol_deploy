
# 🥗 Salad Protocol - AI Nutritionist Chatbot

A comprehensive AI-powered nutritionist chatbot with food image recognition, built with **Next.js frontend** and **Python FastAPI backend**.

---

## 🚀 Features

- 🤖 **AI Nutritionist Chat** – Personalized dietary advice using **Gemini API**
- 📸 **Food Image Recognition** – Custom ML model for food identification and calorie estimation
- 🧠 **RAG System** – Knowledge base with **Upstash Vector** for semantic search
- 👤 **User Profiles** – Personalized responses based on age, height, and weight
- 🔥 **Firebase Integration** – Secure user data storage
- 🌿 **Healthy Design** – Wellness-themed UI with a calming green palette

---

## 🧱 Tech Stack

### 🖥️ Frontend
- [Next.js 14](https://nextjs.org/) (App Router)
- TypeScript + React
- Tailwind CSS + `shadcn/ui`
- Responsive Design

### 🧠 Backend
- [FastAPI](https://fastapi.tiangolo.com/)
- Gemini API (chat + embeddings)
- Firebase Firestore
- Upstash Vector (vector DB for RAG)
- Custom food recognition ML model

---

## 📁 Project Structure

```
salad-protocol/
├── frontend/                 # Next.js frontend
│   ├── app/                  # App Router pages
│   │   ├── page.tsx          # Login page
│   │   ├── profile-setup/    # Profile setup flow
│   │   ├── chat/             # Chat interface
│   │   └── api/              # Frontend API routes
│   └── components/           # Reusable UI components
├── backend/                  # FastAPI backend
│   ├── main.py               # Main FastAPI app
│   ├── auth/                 # Authentication logic
│   ├── chat/                 # AI chat logic
│   ├── food/                 # Food analysis endpoints
│   ├── database/             # Firebase integration
│   ├── vector_db/            # Vector store interface
│   └── custom_model/         # Custom ML model API
└── README.md
```

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

- Gemini API Key  
- Firebase project (with Firestore)
- Upstash Vector Database
- Firebase Admin SDK key file (`serviceAccountKey.json`)

---

### 🧠 Backend Setup

```bash
# 1. Navigate to backend
cd backend

# 2. (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure your Firebase service key is placed here
./backend/serviceAccountKey.json

# 5. Run the backend
uvicorn main:app --reload --port 8000
```

#### 🧪 Run the Custom Model API (in a new terminal)

```bash
cd backend/custom_model
python food_recognition_api.py
```

---

### 🌐 Frontend Setup

```bash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies
npm install

# 3. Run development server
npm run dev
```

---

## 🧪 Environment Configuration

### `.env` (Backend)

```env
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=./serviceAccountKey.json
UPSTASH_VECTOR_REST_URL=your_upstash_url
UPSTASH_VECTOR_REST_TOKEN=your_upstash_token
CUSTOM_MODEL_URL=http://localhost:8001
```

### `.env.local` (Frontend)

```env
PYTHON_BACKEND_URL=http://localhost:8000

NEXT_PUBLIC_FIREBASE_API_KEY=your_firebase_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
```

---

## 🧠 Custom Food Recognition Model

Your FastAPI backend integrates a custom image recognition model for nutritional breakdown.

### 📤 Input
- Image file (JPG/PNG)

### 📥 Output (Example)
```json
{
  "food_items": [
    {
      "name": "Fried Rice",
      "calories": 300,
      "protein": 6.0,
      "carbs": 40.0,
      "fat": 10.0,
      "portion_size": "200g"
    }
  ],
  "total_calories": 300,
  "total_protein": 6.0,
  "total_carbs": 40.0,
  "total_fat": 10.0
}
```

> ⚠️ The current model uses mock data — replace with your trained model.

---

## 🧩 System Flow

1. **Login** – User enters a username (creates account if new).
2. **Profile Setup** – User provides personal health data.
3. **Chat** – Ask diet questions, receive AI-powered answers.
4. **Food Analysis** – Upload image and receive nutritional insights.
5. **Knowledge Base** – Ask about food content with RAG-supported answers.

---

## 💡 Key Modules

### 🧠 RAG Knowledge Base
- Embeddings generated via Gemini API
- Semantic similarity search using Upstash Vector
- Retrieved context sent in prompts

### 👤 Personalization
- Recommendations based on profile (age, height, weight)
- Food history saved for future reference

### 🍱 Food Recognition
- Custom ML model for nutrient estimation
- Easily replaceable with production-grade model

---

## 🧑‍💻 Development Notes

- 🔁 All AI responses are context-aware
- 🧪 Firebase stores only structured data (no images, unless you re-enable storage)
- 🧠 Gemini handles both chat and embedding generation

---

## 🤝 Contributing

1. Fork the repo  
2. Create a new feature branch  
3. Implement changes  
4. Test thoroughly  
5. Submit a PR 🚀

---

## 📄 License

MIT License. See [`LICENSE`](./LICENSE) for details.
