
# part1 baseline_model.py

import tensorflow as tf 

# from tensorflow import keras 

import keras

import numpy as np 

import time 

import os 

 

def create_baseline_model(): 

    """ 

    Create a moderately complex CNN for CIFAR-10 classification. 

    This model is intentionally over-parameterized to demonstrate optimization potential. 

     

    Returns: 

        tf.keras.Model: Compiled model ready for training 

    """ 

    model = keras.Sequential([ 

        # TODO: Implement the required layers 
        keras.Input(shape=(32, 32, 3)),

        # Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) 
        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) 
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) 
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Classifier: GlobalAveragePooling2D -> Dropout(0.5) -> Dense(256) -> Dropout(0.3) -> Dense(10) 
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ]) 

     

    model.compile( 

        optimizer='adam', 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy'] 

    ) 

     

    return model 

 

def load_and_preprocess_data(): 

    """ 

    Load and preprocess CIFAR-10 dataset. 

     

    Returns: 

        tuple: (x_train, y_train, x_test, y_test) 

    """ 

    # TODO: Load CIFAR-10 dataset 
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1] range 
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Apply data augmentation for training set 
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ])

    x_train = data_augmentation(x_train)

    return x_train, y_train, x_test, y_test

 

def train_baseline_model(model, x_train, y_train, x_test, y_test): 

    """ 

    Train the baseline model with early stopping and learning rate scheduling. 

     

    Returns: 

        tuple: (model, training_history, training_metrics) 

    """ 

    # TODO: Implement training with callbacks: 
    
    # - EarlyStopping (patience=10) 

    # - ReduceLROnPlateau 

    # - ModelCheckpoint 

    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    model_checkpoint = keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)

    history = model.fit(
        x_train, y_train,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    # Return model, training history, and metrics
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    metrics = {'test_accuracy': test_accuracy, 'test_loss': test_loss}
    return model, history, metrics


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

def plot_training_history(history):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

def plot_confusion_matrix(model, x_test, y_test, class_names):
    """绘制混淆矩阵"""
    y_pred = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

def plot_sample_predictions(model, x_test, y_test, class_names, num_samples=5):
    """展示随机样本预测"""
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        idx = random.randint(0, len(x_test)-1)
        img, label = x_test[idx], y_test[idx][0]
        pred = model.predict(img.reshape(1, 32, 32, 3)).argmax()
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(f"True: {class_names[label]}\nPred: {class_names[pred]}")
        plt.axis('off')
    plt.savefig("sample_predictions.png")
    plt.show()


if __name__ == "__main__": 

    # Load data 

    x_train, y_train, x_test, y_test = load_and_preprocess_data() 

     

    # Create and train baseline model 

    model = create_baseline_model() 

    model, history, metrics = train_baseline_model(model, x_train, y_train, x_test, y_test) 

     

    # Save baseline model 

    model.save('baseline_model.keras') 
    # 转换为TFLite（关键修改点）
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 添加量化优化（减小模型体积）
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # 确保兼容性
    
    tflite_model = converter.convert()
    
    # 保存TFLite模型
    with open('baseline_model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    # 绘制训练曲线
    plot_training_history(history)

    # CIFAR-10 类别名
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # 绘制混淆矩阵
    plot_confusion_matrix(model, x_test, y_test, class_names)

    # 展示部分预测结果
    plot_sample_predictions(model, x_test, y_test, class_names)

    print(f"Baseline model parameters: {model.count_params():,}") 

    print(f"Baseline test accuracy: {metrics['test_accuracy']:.4f}") 

