import tensorflow as tf
import os
import numpy as np
 

def convert_to_tflite(model_path, quantize=False):

    """

    Convert TensorFlow model to TensorFlow Lite format.

   

    Args:

        model_path (str): Path to saved TensorFlow model

        quantize (bool): Whether to apply quantization

       

    Returns:

        bytes: TensorFlow Lite model data

    """

    # TODO: Load the saved model
    model = tf.keras.models.load_model(model_path)
    # TODO: Create TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model) 
    # TODO: Apply quantization if requested
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # TODO: Convert model and return tflite data
    tflite_model = converter.convert()
    return tflite_model

 

def analyze_model_size(tf_model_path, tflite_model_data):

    """

    Compare model sizes between TensorFlow and TensorFlow Lite.

   

    Args:

        tf_model_path (str): Path to TensorFlow model

        tflite_model_data (bytes): TensorFlow Lite model data

    """

    # TODO: Calculate file sizes and compression ratio
    tf_model_size = os.path.getsize(tf_model_path)
    tflite_model_size = len(tflite_model_data)
    # TODO: Print comparison results
    print(tf_model_size,tflite_model_size,tflite_model_size/tf_model_size)


 

def test_tflite_accuracy(tflite_model_data, x_test, y_test):

    """

    Test TensorFlow Lite model accuracy.

   

    Args:

        tflite_model_data (bytes): TensorFlow Lite model

        x_test, y_test: Test data

       

    Returns:

        float: Test accuracy

    """

    # TODO: Create TensorFlow Lite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_data)
    interpreter.allocate_tensors()
    # TODO: Run inference on test data
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # TODO: Calculate and return accuracy
    correct_predictions = 0
    for i in range(len(x_test)):
        input_data = np.expand_dims(x_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(output_data)

        if predicted_label == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_test)
    return accuracy

 
def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1] range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape data for CNN input (add channel dimension)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train, x_test, y_test)

if __name__ == "__main__":

    # Convert model without quantization

    tflite_model = convert_to_tflite('/Users/bbfish/develop/assignment1/tensorflow/mnist_cnn_model.keras', quantize=False)

   

    # Convert model with quantization

    tflite_quantized_model = convert_to_tflite('/Users/bbfish/develop/assignment1/tensorflow/mnist_cnn_model.keras', quantize=True)

   

    # Save TensorFlow Lite models

    with open('mnist_model.tflite', 'wb') as f:

        f.write(tflite_model)

   

    with open('mnist_model_quantized.tflite', 'wb') as f:

        f.write(tflite_quantized_model)

   

    # Analyze model sizes

    analyze_model_size('/Users/bbfish/develop/assignment1/tensorflow/mnist_cnn_model.keras', tflite_model)

    analyze_model_size('/Users/bbfish/develop/assignment1/tensorflow/mnist_cnn_model.keras', tflite_quantized_model)

   

    # Test accuracy of converted models

    # (You'll need to load test data again)

    x_train, y_train, x_test, y_test = load_and_preprocess_data()

   

    tflite_accuracy = test_tflite_accuracy(tflite_model, x_test, y_test)

    tflite_quantized_accuracy = test_tflite_accuracy(tflite_quantized_model, x_test, y_test)

   

    print(f"TensorFlow Lite accuracy: {tflite_accuracy:.4f}")

    print(f"TensorFlow Lite quantized accuracy: {tflite_quantized_accuracy:.4f}")

