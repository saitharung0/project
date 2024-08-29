import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Hyperparameters
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100
epochs = 10

# Input placeholders (replaced by Input layers in TensorFlow 2.x)
inputs = tf.keras.Input(shape=(784,))

# Define the layers
hidden_1_layer = tf.keras.layers.Dense(n_nodes_hl1, activation='relu')(inputs)
hidden_2_layer = tf.keras.layers.Dense(n_nodes_hl2, activation='relu')(hidden_1_layer)
hidden_3_layer = tf.keras.layers.Dense(n_nodes_hl3, activation='relu')(hidden_2_layer)
output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(hidden_3_layer)

# Build the model
model = tf.keras.Model(inputs=inputs, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy}')
