import pickle
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from .model import build_model, get_metrics
from .utils import load_data

def preprocess_data(data, labels):
    """
    Prepares data for training: converts to numpy, encodes labels, and splits.
    """
    data = np.asarray(data)
    unique_labels = np.unique(labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_to_id[l] for l in labels]
    
    num_classes = len(unique_labels)
    categorical_labels = to_categorical(numeric_labels, num_classes)
    
    return train_test_split(data, categorical_labels, test_size=0.2, shuffle=True, stratify=categorical_labels), unique_labels

def train(data_path, model_save_path):
    """
    Main training function.
    """
    data, labels = load_data(data_path)
    (x_train, x_test, y_train, y_test), unique_labels = preprocess_data(data, labels)
    
    input_shape = x_train.shape[1]
    num_classes = len(unique_labels)
    
    # Save labels for inference
    labels_path = os.path.join(os.path.dirname(model_save_path), 'labels.pickle')
    with open(labels_path, 'wb') as f:
        pickle.dump(unique_labels, f)
    print(f"Labels saved to {labels_path}")
    
    model = build_model(input_shape, num_classes)
    
    print("Starting training...")
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    get_metrics(model, x_test, y_test)
    
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model, history, (x_test, y_test, unique_labels)
