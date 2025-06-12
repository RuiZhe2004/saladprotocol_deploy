# train_inference_engine.py
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd  # For working with dataframes
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report  # For more detailed evaluation
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.pipeline import Pipeline  # for constructing pipelines
from sklearn.compose import ColumnTransformer  # For applying different transformations to different columns
from sklearn.model_selection import cross_val_score, GridSearchCV  # Added GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier  # Neural Network
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction

# 1. Define the Rule Structure
# This represents the kinds of rules your system will use (Reduced for generalization)

RULE_TYPES = [
    "weight_loss",
    "muscle_gain",
    "general_health",
    "specific_nutrient_recommendation",
    "ask_for_more_information" #Kept general purpose classes
]

# 2. Create a Synthetic Training Dataset (Replace with real data)
# This simulates data you would collect about users and queries.

data = {
    "message": [
        "I want to lose weight quickly",
        "How can I build muscle as a beginner?",
        "What are some good sources of fiber?",
        "I think I'm low on iron; what should I eat?",
        "I'm overwhelmed with all the nutrition info!",
        "Best foods for a high-protein, low-carb diet?",
        "What should I eat to manage my type 2 diabetes?",
        "Foods to lower cholesterol?",
        "I'm pregnant; what should I be eating more of?",
        "How many calories should I eat to lose weight a week",
        "Is HIIT or running better for weight loss?",
        "What's a good workout routine for building muscle at home?",
        "Should I take a multivitamin?",
        "I have kidney disease what should I eat",
        "Any recommendations for healthy breakfast options?",
        "Can you suggest a meal plan for someone with diabetes?",
        "Best snacks for heart health?",
        "What vitamins are crucial during pregnancy?",
        "Foods to increase breast milk supply?",
        "I have low blood pressure what should i eat?",
        "Does coffee count as water for hydration?",
        "What's the deal with intermittent fasting?",
        "What are good vegan protein sources?",
        "How to get more sleep and lose weight?",
        "Can i increase my testosterone levels to build muscle fast",
        "My skin is very dry what should i eat",
        "Should i eat more vegetables",
        "What foods are high in vitamin C?",   #More data points
        "Suggest a workout for building muscle",
        "Advice on a diet for type 2 diabetes",
        "What's the best way to lower my LDL?",
    ],
    "user_age": [30, 25, 40, 60, 35, 28, 55, 45, 28, 32, 35, 27, 50, 62, 38, 58, 47, 31, 29, 52, 33, 36, 29, 31, 26, 25, 40, 33, 40, 56, 30],
    "user_weight": [80, 70, 65, 90, 75, 68, 85, 72, 60, 82, 78, 71, 88, 95, 67, 75, 83, 63, 58, 92, 73, 69, 61, 65, 75, 64, 76, 70, 65, 81, 80],
    "user_height": [175, 180, 165, 170, 178, 172, 182, 168, 162, 177, 183, 174, 171, 166, 169, 176, 185, 164, 159, 179, 181, 173, 163, 167, 175, 160, 171, 170, 168, 180, 175],
    "target_rule": [
        "weight_loss",
        "muscle_gain",
        "general_health",
        "specific_nutrient_recommendation",
        "ask_for_more_information",
        "specific_nutrient_recommendation",
        "general_health",  # Simplified from diabetes_management
        "general_health",  # Simplified from heart_health
        "general_health",  # Simplified from pregnancy_nutrition
        "weight_loss",
        "weight_loss",
        "muscle_gain",
        "general_health",
        "specific_nutrient_recommendation",
        "general_health",
        "general_health",  # Simplified from diabetes_management
        "general_health",  # Simplified from heart_health
        "general_health",  # Simplified from pregnancy_nutrition
        "general_health",  # Simplified from pregnancy_nutrition
        "specific_nutrient_recommendation",
        "general_health",
        "general_health",
        "specific_nutrient_recommendation",
        "weight_loss",
        "muscle_gain",
        "general_health",
        "general_health",
        "specific_nutrient_recommendation",
        "muscle_gain",
        "general_health",  # Simplified from diabetes_management
        "general_health"   # Simplified from heart_health
    ]
}

# Ensure all lists have the same length
max_len = max(len(v) for v in data.values())
for key, value in data.items():
    if len(value) < max_len:
        data[key] += [value[-1]] * (max_len - len(value))  # Pad with the last element

df = pd.DataFrame(data)

# Check for NaN values early
if df.isnull().any().any():
    print("WARNING: NaN values detected in training DataFrame! Filling with 0.")
    df = df.fillna(0)

# Print data types of the DataFrame
print("DataFrame data types before feature engineering:\n", df.dtypes)

# 3. Feature Engineering
# Convert text data into numerical features for the model.

# a. TF-IDF for Messages
tfidf_vectorizer = TfidfVectorizer(max_features=50)  # Limit features to prevent overfitting
tfidf_matrix = tfidf_vectorizer.fit_transform(df["message"])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df = pd.concat([df, tfidf_df], axis=1)

# b. BMI (Body Mass Index)
df["bmi"] = df["user_weight"] / ((df["user_height"] / 100) ** 2)

# e. Rule Type Encoding
rule_type_map = {rule: i for i, rule in enumerate(RULE_TYPES)}
df["target_rule_encoded"] = df["target_rule"].map(rule_type_map)

# 4. Prepare Features (X) and Target (y)
# Select the features to use for training.

numerical_features = ["user_age", "bmi"]
text_feature = "message" #Added text_feature for the transformer

# 5. Create preprocessor
# Corrected ColumnTransformer definition
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('text', TfidfVectorizer(max_features=50), text_feature) # Apply TfidfVectorizer
], remainder='drop') #Drop the remaining features instead of passing them through.

X = df[numerical_features + [text_feature]] #Corrected features for model
y = df["target_rule_encoded"]

# 5. Split Data into Training and Testing Sets
from collections import Counter
from sklearn.model_selection import train_test_split

class_counts = Counter(y)
stratify = True

# Print data types of X before split
print("Data types of X before split:\n", X.dtypes)

# Stratify only if there are enough samples per class
min_samples = 2  # Minimum samples needed per class for stratification
if all(count >= min_samples for count in class_counts.values()):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    stratify = True
else:
    print("Skipping stratification due to insufficient samples in some classes.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    stratify = False

# 6. Applying SMOTE: Skip if a class has <=2 samples
print(f"Before SMOTE: {Counter(y_train)}")
class_counts = Counter(y_train)
if any(count <= 2 for count in class_counts.values()):  # Check for classes with too few samples
    print("Skipping SMOTE because one or more classes have very few samples.")
    X_resampled, y_resampled = X_train, y_train
    apply_smote = False #To be used later
else:
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {Counter(y_resampled)}")
    apply_smote = True

# Print data types of X_resampled after SMOTE
print("Data types of X_resampled after SMOTE:\n", X_resampled.dtypes)

# Create Pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', RandomForestClassifier(random_state=42))])  # Simplified model definition

# 7. Hyperparameter Tuning using GridSearchCV
# In train_inference_engine.py
param_grid = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__min_samples_split': [2, 5, 10, 15],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0)  # Reduced verbosity
grid_search.fit(X_resampled, y_resampled)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_) #Best Parameters
print("Pipeline before saving:", grid_search.best_estimator_)  #Print the entire pipeline, it needs to show fitted
#Inspect TfidfVectorizer to verify if it's fitted
print("TfidfVectorizer vocabulary:", grid_search.best_estimator_.named_steps['preprocessor'].transformers_[1][1].vocabulary_)

# 8. Evaluate the Model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Adjust classification report to handle cases where not all classes are present
unique_classes_test = np.unique(y_test)
target_names_present = [RULE_TYPES[i] for i in unique_classes_test]

# Check that the test y has any values before doing the test
if len(target_names_present)>0:
  # Fix the classification report error
  labels = np.unique(np.concatenate([y_test, y_pred]))  # Get all unique labels
  target_names_present = [RULE_TYPES[i] for i in labels]  # Names for all labels

  print(classification_report(y_test, y_pred, target_names=target_names_present, zero_division=0, labels=labels))
else:
  print("The `y_test` labels has no values")

# 9. Save the Trained Model

model_path = "inference_engine_model.joblib"
joblib.dump({
    'pipeline': grid_search.best_estimator_, #Save the entire Pipeline
    'rule_types': RULE_TYPES
}, model_path)
print(f"Inference engine model saved to {model_path}")

print(df["target_rule"].value_counts())
print(df[["message", "user_age", "bmi"]].head())  # Changed the printed columns
print("Training complete!")