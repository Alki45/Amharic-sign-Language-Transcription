import argparse
import os
from src.data_processing import process_data_directory
from src.utils import save_data
from src.train import train
from src.inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="Amharic Sign Language Transcription Pipeline")
    parser.add_argument("mode", choices=["process", "train", "test"], help="Operation mode: 'process' for landmark extraction, 'train' for model training, 'test' for real-time inference.")
    parser.add_argument("--data_dir", default="data/raw", help="Directory containing raw images (for 'process' mode).")
    parser.add_argument("--pickle_path", default="data/processed/data.pickle", help="Path to save/load processed landmarks.")
    parser.add_argument("--model_path", default="models/model.h5", help="Path to save/load the trained model.")
    parser.add_argument("--labels_path", default="models/labels.pickle", help="Path to save/load the genre labels.")

    args = parser.parse_args()

    if args.mode == "process":
        print(f"Processing data from {args.data_dir}...")
        data, labels = process_data_directory(args.data_dir)
        
        # Ensure processed directory exists
        os.makedirs(os.path.dirname(args.pickle_path), exist_ok=True)
        save_data(data, labels, args.pickle_path)
        
    elif args.mode == "train":
        print(f"Training model using data from {args.pickle_path}...")
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        train(args.pickle_path, args.model_path)
        
    elif args.mode == "test":
        print(f"Running real-time inference using {args.model_path}...")
        run_inference(args.model_path, args.labels_path)

if __name__ == "__main__":
    main()
