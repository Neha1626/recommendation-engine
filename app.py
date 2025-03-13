import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Global variables to store model and data (loaded lazily)
model = None
user_item_matrix = None
n_users = None
n_items = None

print("Starting app initialization...")

def load_resources():
    global model, user_item_matrix, n_users, n_items
    if model is None or user_item_matrix is None:
        print("Loading model...")
        model = load_model("recommendation_model.h5", compile=False)
        print("Model loaded successfully.")
        print("Loading user_item_matrix...")
        user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0).astype(np.float32)
        print("user_item_matrix loaded successfully.")
        n_users = user_item_matrix.shape[0]
        n_items = user_item_matrix.shape[1]
        print(f"n_users: {n_users}, n_items: {n_items}")
        if hasattr(model, 'optimizer'):
            del model.optimizer  # Free up memory

@app.route('/')
def home():
    print("Handling home route...")
    load_resources()
    print("Resources loaded for home route.")
    return render_template('index.html', n_users=n_users)

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    print(f"Handling recommend route for user_id: {user_id}")
    load_resources()
    if user_id < 0 or user_id >= n_users:
        return jsonify({"error": "Invalid user ID"}), 400

    item_ids = np.arange(n_items)
    user_ids = np.array([user_id] * n_items)
    predictions = model.predict([user_ids, item_ids], batch_size=64, verbose=0)

    user_ratings = user_item_matrix.iloc[user_id].values
    prediction_scores = predictions.flatten()
    unrated_mask = user_ratings == 0
    unrated_scores = prediction_scores[unrated_mask]
    top_items_indices = np.argsort(unrated_scores)[-5:][::-1]
    top_items = np.where(unrated_mask)[0][top_items_indices]

    recommended_items = user_item_matrix.columns[top_items].tolist()
    return jsonify({"user_id": user_id, "recommended_items": recommended_items})

@app.route('/recommend', methods=['POST'])
def recommend_form():
    print("Handling recommend_form route...")
    load_resources()
    user_id = int(request.form['user_id'])
    print(f"User ID: {user_id}")
    if user_id < 0 or user_id >= n_users:
        return render_template('index.html', error="Invalid user ID. Please enter a number between 0 and " + str(n_users-1), n_users=n_users)

    item_ids = np.arange(n_items)
    user_ids = np.array([user_id] * n_items)
    predictions = model.predict([user_ids, item_ids], batch_size=64, verbose=0)

    user_ratings = user_item_matrix.iloc[user_id].values
    prediction_scores = predictions.flatten()
    unrated_mask = user_ratings == 0
    unrated_scores = prediction_scores[unrated_mask]
    top_items_indices = np.argsort(unrated_scores)[-5:][::-1]
    top_items = np.where(unrated_mask)[0][top_items_indices]

    recommended_items = user_item_matrix.columns[top_items].tolist()
    return render_template('index.html', user_id=user_id, recommendations=recommended_items, n_users=n_users)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)