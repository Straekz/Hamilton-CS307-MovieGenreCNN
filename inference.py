import os
import ast
import argparse
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from train import I_V3Movie, calculate_mean_std, MoviePostersDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALL_GENRES = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

class TestDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_file)
        self.labels_df['encoded_genres'] = self.labels_df['encoded_genres'].apply(ast.literal_eval)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        # label = torch.tensor(row['encoded_genres'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, row['filename'], row['encoded_genres']

def inference(image_folder, labels_path, model_path):
    # Load labels
    labels_df = pd.read_csv(labels_path)
    print(f"CSV Columns: {labels_df.columns}")

    # Load model
    num_genres = len(ast.literal_eval(labels_df['encoded_genres'].iloc[0]))
    model = I_V3Movie(num_genres=num_genres).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    norm_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    temp = MoviePostersDataset("data/posters", "data/preprocessed.csv", transform=norm_transform)
    mean, std = calculate_mean_std(temp)

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = TestDataset(image_folder, labels_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    results = []

    with torch.no_grad():
        for images, filenames, actual_genres in dataloader:
            images = images.to(DEVICE)
            outputs = torch.sigmoid(model(images))  # Multi-label classification
            predictions = (outputs > 0.5).int().cpu().tolist()

            for filename, pred, actual in zip(filenames, predictions, actual_genres):
                results.append({
                    "filename": filename,
                    "predicted_genres": pred,
                    "actual_genres": actual.tolist()
                })

    # Calculate accuracy
    correct = sum(result['predicted_genres'] == result['actual_genres'] for result in results)
    accuracy = correct / len(results) if results else 0.0
    print(f"Accuracy: {accuracy:.2%}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("inference_results.csv", index=False)
    print("Inference results saved to 'inference_results.csv'")

def main():
    parser = argparse.ArgumentParser(description="Inference script for genre classification.")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to folder containing images.")
    parser.add_argument('--labels', type=str, required=True, help="Path to CSV file containing labels.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file.")

    args = parser.parse_args()
    inference(args.image_folder, args.labels, args.model_path)

if __name__ == "__main__":
    main()
