import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Dense, Reshape
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Load the user-item matrix
print("Loading user-item matrix...")
user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0)
user_item_matrix = user_item_matrix.astype(np.float32)

print("Matrix shape:", user_item_matrix.shape)

# Prepare data
n_users = user_item_matrix.shape[0]
n_items = user_item_matrix.shape[1]
ratings = user_item_matrix.values

print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")

# Create training data (only use non-zero ratings for training)
print("\nPreparing training data...")
user_ids, item_ids = np.where(ratings > 0)
ratings_flat = ratings[user_ids, item_ids]

print(f"Training data shape: {ratings_flat.shape}")

# Build the model
print("\nBuilding the model...")
embedding_size = 50

user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(n_users, embedding_size)(user_input)
item_embedding = Embedding(n_items, embedding_size)(item_input)

user_vec = Reshape([embedding_size])(user_embedding)
item_vec = Reshape([embedding_size])(item_embedding)

dot_product = Dot(axes=1)([user_vec, item_vec])
output = Dense(1, activation='linear')(dot_product)

model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())  # Explicitly use MeanSquaredError

print("\nModel summary:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    [user_ids, item_ids],
    ratings_flat,
    epochs=10,
    batch_size=64,
    verbose=1,
    validation_split=0.2
)

# Save the model
print("\nSaving the model...")
model.save("recommendation_model.h5", save_format='h5')  # Explicitly use h5 format

print("\nModel trained and saved!")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')
plt.close()

print("\nTraining history plot saved as 'training_history.png'")