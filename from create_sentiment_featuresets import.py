import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import os

lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_classes = 2
batch_size = 32
hm_epochs = 10
input_dim = 2638  # Example input dimension

class SentimentModel(tf.keras.Model):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(n_nodes_hl1, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(n_nodes_hl2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')

    def call(self, inputs):
        x = self.hidden_1(inputs)
        x = self.hidden_2(x)
        return self.output_layer(x)

def train_neural_network():
    model = SentimentModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    X_train = np.random.randn(20000, input_dim)
    y_train = np.eye(n_classes)[np.random.choice(n_classes, 20000)]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=hm_epochs)

    # Save the model
    model.save('./sentiment_model.h5')

def use_neural_network(input_data):
    with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    current_words = word_tokenize(input_data.lower())
    current_words = [lemmatizer.lemmatize(i) for i in current_words]
    features = np.zeros(len(lexicon))

    for word in current_words:
        if word.lower() in lexicon:
            index_value = lexicon.index(word.lower())
            features[index_value] += 1

    features = np.array([features])

    if os.path.exists('./sentiment_model.h5'):
        model = tf.keras.models.load_model('./sentiment_model.h5')
    else:
        print("No model found. Train the model first.")
        return

    result = model.predict(features)
    sentiment = np.argmax(result)
    if sentiment == 0:
        print('Positive:', input_data)
    else:
        print('Negative:', input_data)

# Train the model if needed (comment out if the model is already trained)
train_neural_network()

# Use the neural network for prediction
use_neural_network("He's an idiot and a jerk.")
use_neural_network("This was the best store I've ever seen.")
