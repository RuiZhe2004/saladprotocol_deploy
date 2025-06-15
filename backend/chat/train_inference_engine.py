# train_inference_engine.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import logging
import re
from collections import Counter
from imblearn.over_sampling import SMOTE

# Additions for Word Embeddings
import spacy
#python -m spacy download en_core_web_md
nlp = spacy.load("en_core_web_md")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Define the Rule Structure
RULE_TYPES = [
    "weight_loss", "muscle_gain", "general_health", "specific_nutrient_recommendation",
    "ask_for_more_information", "hydration_recommendation", "sleep_improvement",
    "stress_management_nutrition", "energy_boosting_foods", "skin_health_nutrition",
    "digestive_health", "immune_boosting", "bone_health", "eye_health_nutrition",
    "brain_health", "hair_health_nutrition", "pregnancy", "diabetes",
    "high_cholesterol", "low_blood_pressure", "kidney_issues", "anxiety",
]

# 2. Create/Load Training Data
data = {
    "message": [
        "I want to lose weight quickly",
        "How can I build muscle as a beginner?",
        "What are some good sources of fiber?",
        "I think I'm low on iron; what should I eat?",
        "I'm overwhelmed with all the nutrition info!",
        "Best foods for a high-protein, low-carb diet?",
        "What should I eat to manage my blood sugar?",
        "Foods to lower cholesterol?",
        "I'm trying to eat healthier during pregnancy; what should I focus on?",
        "How many calories should I eat to lose weight a week",
        "Is HIIT or running better for weight loss?",
        "What's a good workout routine for building muscle at home?",
        "Should I take a multivitamin?",
        "I have kidney issues; what foods should I limit?",
        "Any recommendations for healthy breakfast options?",
        "Can you suggest a meal plan for someone with blood sugar concerns?",
        "Best snacks for heart health?",
        "What vitamins are crucial when expecting?",
        "Foods to increase milk production?",
        "I have low blood pressure what should i eat?",
        "Does coffee count as water for hydration?",
        "What's the deal with intermittent fasting?",
        "What are good vegan protein sources?",
        "How to get more sleep and lose weight?",
        "Can i increase my testosterone levels to build muscle fast",
        "My skin is very dry what should i eat",
        "Should i eat more vegetables",
        "What foods are high in vitamin C?",
        "Suggest a workout for building muscle",
        "Advice on a diet for blood sugar control",
        "What's the best way to lower my LDL?",
        "I feel dehydrated, what should I drink?",
        "How can I improve my sleep?",
        "Stress eating is ruining my diet, help!",
        "What foods give you the most energy?",
        "My skin is breaking out, any dietary tips?",
        "I'm always bloated, what should I eat?",
        "How can I boost my immune system with food?",
        "What foods are good for strong bones?",
        "What can I eat to protect my vision?",
        "Foods to improve memory and focus?",
        "What should I eat for healthier hair?",
        "How many glasses of water should I drink a day?",
        "Tips for a better night's sleep?",
        "Foods that help with stress and anxiety?",
        "What's a quick and healthy energy boost?",
        "Foods to reduce acne and improve skin?",
        "How to reduce bloating naturally?",
        "Best foods to prevent colds?",
        "Nutrients for bone density?",
        "Best foods for eye strain?",
        "Foods for brain fog and concentration?",
        "Diet tips for thicker, stronger hair?",
        "I am gaining weight uncontrollably",
        "Give me foods to eat for lunch",
        "How can i eat to improve my health generally",
        "I want to stop eating fast food. How do I do that",
        "I am very tired recently",
        "I want to get better skin health and lose weight",
        "I have been feeling quite anxious recently",
        "I would like to eat more healthy",
        "I have low bone density what should i do"
    ],
    "user_age": [
        30, 25, 40, 60, 35, 28, 55, 45, 28, 32,
        35, 27, 50, 62, 38, 58, 47, 31, 29, 52,
        33, 36, 29, 31, 26, 25, 40, 33, 40, 56,
        30, 24, 42, 39, 22, 37, 48, 51, 26, 33,
        41, 27, 34, 43, 29, 36, 49, 53, 24, 31,
        38, 25, 43, 30, 63, 47, 27, 35, 40, 50
    ],
    "user_weight": [
        80, 70, 65, 90, 75, 68, 85, 72, 60, 82,
        78, 71, 88, 95, 67, 75, 83, 63, 58, 92,
        73, 69, 61, 65, 75, 64, 76, 70, 65, 81,
        80, 68, 74, 71, 59, 73, 86, 94, 62, 79,
        66, 70, 63, 77, 60, 67, 84, 91, 61, 75,
        72, 100, 72, 83, 90, 70, 60, 83, 65, 75
    ],
    "user_height": [
        175, 180, 165, 170, 178, 172, 182, 168, 162, 177,
        183, 174, 171, 166, 169, 176, 185, 164, 159, 179,
        181, 173, 163, 167, 175, 160, 171, 170, 168, 180,
        175, 173, 169, 166, 161, 172, 184, 170, 163, 178,
        167, 171, 165, 179, 160, 168, 186, 172, 162, 176,
        174, 178, 173, 165, 170, 165, 160, 170, 168, 170
    ],
    "target_rule": [
        "weight_loss", "muscle_gain", "general_health",
        "specific_nutrient_recommendation", "ask_for_more_information",
        "specific_nutrient_recommendation", "diabetes", "high_cholesterol",
        "pregnancy", "weight_loss", "weight_loss", "muscle_gain",
        "general_health", "kidney_issues", "general_health", "diabetes",
        "high_cholesterol", "pregnancy", "general_health", "low_blood_pressure",
        "hydration_recommendation", "general_health", "specific_nutrient_recommendation",
        "sleep_improvement", "muscle_gain", "skin_health_nutrition", "general_health",
        "specific_nutrient_recommendation", "muscle_gain", "diabetes", "high_cholesterol",
        "hydration_recommendation", "sleep_improvement", "stress_management_nutrition",
        "energy_boosting_foods", "skin_health_nutrition", "digestive_health", "immune_boosting",
        "bone_health", "eye_health_nutrition", "brain_health", "hair_health_nutrition",
        "hydration_recommendation", "sleep_improvement", "stress_management_nutrition",
        "energy_boosting_foods", "skin_health_nutrition", "digestive_health", "immune_boosting",
        "bone_health", "eye_health_nutrition", "brain_health", "weight_loss",
        "general_health", "general_health", "general_health", "energy_boosting_foods",
        "skin_health_nutrition", "anxiety", "general_health", "bone_health"
    ]
}

# Ensure all lists have the same length
max_len = max(len(v) for v in data.values())
for key, value in data.items():
    if len(value) < max_len:
        data[key] += [value[-1]] * (max_len - len(value))  # Pad with the last element

# Target: Have at least 5 samples of each
target_min_samples = 5

#Oversample rare classes BEFORE creating dataframe
for rule_type in RULE_TYPES:
    rule_indices = [i for i, rule in enumerate(data['target_rule']) if rule == rule_type]
    samples_needed = max(0, target_min_samples - len(rule_indices))  # Ensure non-negative

    if samples_needed > 0 and len(rule_indices) > 0:
        for _ in range(samples_needed):
            index_to_duplicate = rule_indices[0] # Duplicate the first available index
            for key in data:
                data[key].append(data[key][index_to_duplicate])

# Re-calculate target counts after oversampling, before dataframe creation
target_counts = Counter(data['target_rule'])
print("Target Rule Counts after oversampling:\n", target_counts)

df = pd.DataFrame(data)

# Check for NaN values early
if df.isnull().any().any():
    logger.warning("WARNING: NaN values detected in training DataFrame! Filling with 0.")
    df = df.fillna(0)

# Print data types of the DataFrame
logger.info(f"DataFrame data types before feature engineering:\n{df.dtypes}")

# 3. Feature Engineering
# Calculate BMI *before* any text cleaning
df["bmi"] = df["user_weight"] / ((df["user_height"] / 100) ** 2)
print("DataFrame Columns after BMI Calculation:\n", df.columns)

# **RULE TYPE ENCODING - MOVED UP**
rule_type_map = {rule: i for i, rule in enumerate(RULE_TYPES)}
print("rule_type_map:\n", rule_type_map) #Debugging

#Print unique target rules
unique_target_rules = df["target_rule"].unique()
print("Unique Target Rules in DataFrame:\n", unique_target_rules) #Debugging

#Perform mapping, handling missing keys
df["target_rule_encoded"] = df["target_rule"].map(rule_type_map)

#Verify if all mappings were successful
print("Number of NaN values in target_rule_encoded:\n", df["target_rule_encoded"].isnull().sum())

#Check why the mapping has errors
if df["target_rule_encoded"].isnull().sum() > 0:
  #Investigate what is causing a specific row to have errors
  problem_row = df[df["target_rule_encoded"].isnull()]
  print("Problematic Rows:\n", problem_row)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    STOPWORDS = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
        "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
        "weren", "won", "wouldn"
    ])
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

df['message_cleaned'] = df['message'].apply(clean_text)

# Function to get the vector representation of the message
def get_message_embedding(message):
    doc = nlp(message)
    return doc.vector

# Apply the function to create embeddings
df['message_embedding'] = df['message_cleaned'].apply(get_message_embedding)

# 4. Prepare Features (X) and Target (y)
numerical_features = ["user_age", "bmi"]
print("Numerical Features being used:", numerical_features)  # Debugging
print("DataFrame Columns:\n", df.columns) # Debugging
text_feature = 'message_embedding'  # Use embeddings

# Stack embeddings into a 2D array
X_text = np.stack(df['message_embedding'].to_numpy())
X_numerical = df[numerical_features].to_numpy()

# Combine numerical features with text embeddings
X = np.concatenate((X_numerical, X_text), axis=1)
y = df["target_rule_encoded"].to_numpy()  # Convert to numpy array

# 5. Split Data into Training, Validation, and Testing Sets

try:
  X_train, X_temp, y_train, y_temp = train_test_split(
      X, y, test_size=0.5, random_state=42, stratify=y # test size increased
  )
except ValueError as e:
    print(f"Stratified split failed: {e}.  Disabling stratification.")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=42 #Removing stratify
    )

try:
  X_val, X_test, y_val, y_test = train_test_split(
      X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
  )
except ValueError as e:
    print(f"Stratified split failed: {e}.  Disabling stratification.")
    X_val, X_test, y_val, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42 #Removing stratify
  )

# 6. Addressing Data Imbalance
min_samples_class = 1 if len(np.bincount(y_train)) == 0 else np.min(np.bincount(y_train))#Added in case there are no points
k_neighbors = max(1, min(5, min_samples_class - 1)) #Ensure k_neighbors is at least 1

# Skip SMOTE if smallest class has fewer than 2 samples
if min_samples_class < 2:
    print("Skipping SMOTE: Smallest class has fewer than 2 samples. No model is trained")
    X_resampled, y_resampled = X_train, y_train
    y_pred = np.array([]) # Setting value to be empty array.
    best_model = None #Adding the best_model to be None as well
else:
    smote = SMOTE(random_state=42, k_neighbors= k_neighbors ) # k_neighbors must be smaller than the number of samples
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # 7. Model Training
    model = SVC(random_state=42, class_weight='balanced') # class_weight='balanced' addresses imbalance
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf'] #, 'linear', 'poly', 'sigmoid'] #Removed to reduce training time
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_macro', verbose=2)
    grid_search.fit(X_resampled, y_resampled)

    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    y_pred = best_model.predict(X_test) # Now assigned here after Gridsearch is complete

# 8. Evaluate the Model
# Adjust classification report to handle cases where not all classes are present
unique_classes_test = np.unique(y_test)
target_names_present = [RULE_TYPES[i] for i in unique_classes_test]

# Check that the test y has any values before doing the test
if len(target_names_present)>0 and best_model is not None:
  # Fix the classification report error
  labels = np.unique(np.concatenate([y_test, y_pred])).astype(int)  # Get all unique labels

  target_names_present = [RULE_TYPES[i] for i in labels]  # Names for all labels

  print(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=target_names_present, zero_division=0, labels=labels)}")
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Test Accuracy: {accuracy}") #Added here as well
else:
  print("The `y_test` labels has no values, or the models did not get trained, skipping the evalution and classification report.")

# 9. Save the Model
model_path = "inference_engine_model.joblib"

# only if there is SMOTE being run
if best_model is not None:
  joblib.dump({
      'model': best_model, #Changed from Pipeline
      'rule_types': RULE_TYPES
  }, model_path)

  logger.info(f"Sample Data:\n{df[['message', 'user_age', 'bmi']].head()}")  # Changed the printed columns
  logger.info("Training complete!")
else:
  print("The SMOTE could not be run, the model will not be saved.")