import os
from PIL import Image
import pandas as pd

#checks for corrupted images
def check_images_in_folder(folder_path):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    # Prepare data for pandas DataFrame
    data = []
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            # Try to open and verify the image
            with Image.open(file_path) as img:
                img.verify()  # Verify the image
            data.append({"file_name": file_name, "status": "OK"})
        except Exception as e:
            # If an error occurs, log the file as corrupted
            data.append({"file_name": file_name, "status": "Corrupted", "error": str(e)})
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

# Example usage
folder_path = "data/French/posters"  # Replace with the path to your folder
result_df = check_images_in_folder(folder_path)

# Display the results
print(result_df)

# Optionally save the results to a CSV
result_df.to_csv("data/image_check_results2.csv", index=False)
