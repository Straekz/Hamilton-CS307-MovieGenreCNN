import os
import ast
import argparse
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from train import I_V3Movie, MoviePostersDataset

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
        label = torch.tensor(row['encoded_genres'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label, row['filename']

# Function to calculate match percentage for each row
def calculate_accuracy(row):
    matches = sum([1 if a == b else 0 for a, b in zip(row['Actual'], row['Prediction'])])
    accuracy = (matches / len(row['Actual'])) * 100
    return accuracy

def compare_exact(row):
    if (sum([a == b for a, b in zip(row['Actual'], row['Prediction'])]) == 19):
        return 1
    return 0

def inference(image_folder, labels_path, model_path):
    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Load model
    num_genres = len(ast.literal_eval(labels_df['encoded_genres'].iloc[0]))
    model = I_V3Movie(num_genres=num_genres).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48450906, 0.42860913, 0.39646831], std=[0.34108709, 0.32412509, 0.319536]),
    ])
    dataset = TestDataset(image_folder, labels_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    results = []

    with torch.no_grad():
        for images, labels, filenames in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            predictions = (outputs.sigmoid() > 0.5).int().tolist()
            results.extend(zip(filenames, predictions))

    results_df = pd.DataFrame(results, columns=['Filename', 'Prediction'])
    results_df['Actual'] = labels_df['encoded_genres'].apply(ast.literal_eval)

    # Calculate accuracy
    results_df['Accuracy'] = results_df.apply(calculate_accuracy, axis=1)
    overall_accuracy = results_df['Accuracy'].mean()
    results_df['Correct'] = results_df.apply(compare_exact, axis=1)
    correctness = results_df['Correct'].mean() * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Exact Correctness Percentage: {correctness:.2f}%")

    # Save results to CSV
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
