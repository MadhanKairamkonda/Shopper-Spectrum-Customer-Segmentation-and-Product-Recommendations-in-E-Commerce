# Shopper-Spectrum-Customer-Segmentation-and-Product-Recommendations-in-E-Commerce
Customer Segmentation and Product Recommendation system using RFM analysis, K-Means clustering, and item-based collaborative filtering. Includes end-to-end data preprocessing, EDA, model evaluation, and an interactive Streamlit app for real-time insights.
Problem Statement:
Tourism agencies and travel platforms aim to enhance user experiences by leveraging data to provide personalized recommendations, predict user satisfaction, and classify potential user behavior. This project involves analyzing user preferences, travel patterns, and attraction features to achieve three primary objectives: regression, classification, and recommendation.
This system integrates:

1️⃣ Content-Based Learning

Model: RandomForestRegressor

Uses engineered features:

User average rating

User visit count

Attraction average rating

Attraction popularity

Handles cold-start scenarios

2️⃣ Collaborative Filtering

Custom Matrix Factorization implemented from scratch using NumPy

Trained with Stochastic Gradient Descent (SGD)

Learns latent user and attraction embeddings

3️⃣ Hybrid Model

Final Score:

Final = α × Collaborative Score + (1 − α) × Content Score

Adjustable α parameter

Combines behavioral patterns + item features

This system integrates:

🧠 Machine Learning Pipeline
Data Handling

Train/test split performed before feature engineering (prevents leakage)

Missing values handled safely

High-cardinality text columns removed

Feature Engineering

User_Avg_Rating

User_Total_Visits

Attraction_Avg_Rating

Attraction_Popularity

Encoding

One-hot encoding via pd.get_dummies()

Hyperparameter Tuning

GridSearchCV for both:

RandomForestRegressor

RandomForestClassifier

Balanced class weights used for classification
📊 Model Evaluation
Regression (Rating Prediction)

R² Score: ~0.71

MSE: ~0.27

MAE: ~0.26

Classification (VisitMode)

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Collaborative Filtering

RMSE

MAE

🖥 Streamlit Application

Interactive web interface with:

User selection

Adjustable hybrid weight (α)

Top-N recommendation generation

Cold-start fallback handling
