import os
import uuid
from dotenv import load_dotenv
from upstash_vector import Index, Vector
import google.generativeai as genai
from knowledge_base.loader import load_nutrition_guidelines

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))



def embed_text(text: str):
    """Embeds a given text using the embedding-001 model."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response['embedding']
# Load and flatten your JSON as before
guidelines = load_nutrition_guidelines()

def flatten_json(json_data):
    docs = []
    def recurse(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(f"{prefix} > {k}" if prefix else k, v)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recurse(f"{prefix} [{i}]", item)
        else:
            docs.append(f"{prefix}: {str(obj)}")
    recurse("", json_data)
    return docs

docs = flatten_json(guidelines)

index = Index(url=os.getenv("UPSTASH_VECTOR_REST_URL"), token=os.getenv("UPSTASH_VECTOR_REST_TOKEN"))

vectors = []
for doc in docs:
    print(f"Embedding: {doc[:60]}...")
    embedding = embed_text(doc)
    vectors.append(Vector(id=str(uuid.uuid4()), vector=embedding, metadata={"text": doc[:100]}))

index.upsert(vectors=vectors)

print(f"✅ Successfully embedded and uploaded {len(vectors)} chunks.")
