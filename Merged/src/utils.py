import pickle
import os

def save_data(data, labels, file_path):
    """
    Saves data and labels to a pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Data saved to {file_path}")

def load_data(file_path):
    """
    Loads data and labels from a pickle file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")
    
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    print(f"Data loaded from {file_path}")
    return data_dict['data'], data_dict['labels']
