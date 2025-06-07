import json
from pathlib import Path

def load_nutrition_guidelines():
    file_path = Path(__file__).parent / "nutritionguideline.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data