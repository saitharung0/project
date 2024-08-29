import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load the pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/bert-for-sentiment-analysis/1"  # Replace with an appropriate model URL if needed
model = hub.KerasLayer(model_url, trainable=False)

# Define the preprocessing function
def preprocess_text(text):
    # Simple preprocessing: convert to lowercase and strip
    return text.lower().strip()

# Define the prediction function
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_array = np.array([processed_text])
    predictions = model(text_array)
    sentiment = np.argmax(predictions.numpy())
    return sentiment

# Example usage
sentiment = predict_sentiment("I love this product!")
print("Sentiment:", "Positive" if sentiment == 1 else "Negative")
