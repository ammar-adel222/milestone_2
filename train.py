import os
import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from preprocess import preprocess, TARGET_COLUMN

# =====================================
# LOAD DATASET
# =====================================

file_path = 'train_data.csv'

df = pd.read_csv(file_path)



# =====================================
# PREPROCESSING (shared with test.py)
# =====================================

preprocess(df)

# =====================================
# TARGET COLUMN
# =====================================

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]

# =====================================
# ENCODE TARGET LABELS
# =====================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# SAVE LABEL ENCODER

os.makedirs('saved_models', exist_ok=True)

joblib.dump(
    label_encoder,
    'saved_models/label_encoder.pkl'
)

# =====================================
# IDENTIFY FEATURE TYPES
# =====================================

numeric_features = X.select_dtypes(
    include=['int64', 'float64']
).columns

categorical_features = X.select_dtypes(
    include=['object']
).columns



# =====================================
# PREPROCESSING PIPELINE
# =====================================

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =====================================
# MODELS
# =====================================

# class_weight='balanced' addresses the severe class imbalance
# (Low=83%, Medium=6%, High=11%) by penalizing misclassifications
# of minority classes more heavily.

models_params = {
    'SVM': (
        SVC(random_state=42, class_weight='balanced'),
        {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear']
        }
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [10, 20, None]
        }
    ),
    'KNN': (
        KNeighborsClassifier(n_jobs=-1),
        {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__p': [1, 2]
        }
    )
}

accuracies = []

balanced_accuracies = []

training_times = []

testing_times = []

best_model = None

best_cv_score = 0

best_model_name = ''

# =====================================
# TRAIN AND EVALUATE
# =====================================

for model_name, (model, params) in models_params.items():

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # TRAINING & TUNING
    # Using balanced_accuracy to properly evaluate on imbalanced classes
    grid_search = GridSearchCV(
        pipeline, params, cv=3,
        scoring='balanced_accuracy', n_jobs=-1, verbose=0
    )

    start_train = time.time()

    grid_search.fit(X_train, y_train)

    end_train = time.time()

    train_time = end_train - start_train

    best_pipeline = grid_search.best_estimator_

    # TESTING

    start_test = time.time()

    predictions = best_pipeline.predict(X_test)

    end_test = time.time()

    test_time = end_test - start_test

    # METRICS

    accuracy = accuracy_score(y_test, predictions)

    bal_accuracy = balanced_accuracy_score(y_test, predictions)

    print(f'\n{model_name}')
    print(f'  Accuracy:             {accuracy:.4f}')
    print(f'  CV Balanced Accuracy: {grid_search.best_score_:.4f}')
    print(f'  Confusion Matrix:')
    cm = confusion_matrix(y_test, predictions)
    for row in cm:
        print(f'    {row}')

    accuracies.append(accuracy)

    balanced_accuracies.append(bal_accuracy)

    training_times.append(train_time)

    testing_times.append(test_time)

    # SELECT BEST MODEL USING CV SCORE (not test accuracy)
    # This keeps the test set as a true held-out evaluation

    if grid_search.best_score_ > best_cv_score:

        best_cv_score = grid_search.best_score_

        best_model = best_pipeline

        best_model_name = model_name

# =====================================
# SAVE BEST MODEL
# =====================================

joblib.dump(
    best_model,
    'saved_models/best_model.pkl'
)

print('\n=====================================')

print(f'Best Model: {best_model_name}')

print(f'Best CV Balanced Accuracy: {best_cv_score:.4f}')

print('=====================================')

# =====================================
# BAR GRAPH - BALANCED ACCURACY
# =====================================

model_names = list(models_params.keys())

plt.figure(figsize=(8, 5))

plt.bar(model_names, balanced_accuracies)

plt.title('Balanced Accuracy Comparison')

plt.xlabel('Models')

plt.ylabel('Balanced Accuracy')

plt.savefig('saved_models/balanced_accuracy_chart.png', dpi=150, bbox_inches='tight')

plt.show()

# =====================================
# BAR GRAPH - TRAINING TIME
# =====================================

plt.figure(figsize=(8, 5))

plt.bar(model_names, training_times)

plt.title('Training Time Comparison')

plt.xlabel('Models')

plt.ylabel('Time (seconds)')

plt.savefig('saved_models/training_time_chart.png', dpi=150, bbox_inches='tight')

plt.show()

# =====================================
# BAR GRAPH - TESTING TIME
# =====================================

plt.figure(figsize=(8, 5))

plt.bar(model_names, testing_times)

plt.title('Testing Time Comparison')

plt.xlabel('Models')

plt.ylabel('Time (seconds)')

plt.savefig('saved_models/testing_time_chart.png', dpi=150, bbox_inches='tight')

plt.show()