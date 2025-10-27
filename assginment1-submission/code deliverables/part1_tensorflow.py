import tensorflow as tf
from tensorflow import keras
import numpy as np

def create_model():
    """
    Create the CNN model for MNIST classification.

    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    model = keras.Sequential([
        # Layer 1: Conv2D with 8 filters, 3x3 kernel, ReLU activation
        keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Layer 2: MaxPool2D with 2x2 pool size
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Layer 3: Flatten
        keras.layers.Flatten(),
        # Layer 4: Dense with 16 units, ReLU activation
        keras.layers.Dense(16, activation='relu'),
        # Layer 5: Dense with 10 units (output layer)
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1] range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape data for CNN input (add channel dimension)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train, x_test, y_test)

def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train the model and evaluate performance.

    Args:
        model: Compiled Keras model
        x_train, y_train: Training data
        x_test, y_test: Test data

    Returns:
        tf.keras.callbacks.History: Training history
    """
    # Train for 5 epochs with validation
    history = model.fit(
        x_train, y_train,
        epochs=5,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Return training history for analysis
    return history

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Create and train model
    model = create_model()
    history = train_model(model, x_train, y_train, x_test, y_test)

    # Save the trained model
    model.save('mnist_cnn_model.keras')

    # Evaluate final performance
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")