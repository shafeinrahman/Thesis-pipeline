import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

def generate_mock_dataset():
    print("=" * 60)
    print("Generating Mock DDI Dataset for Testing...")
    print("=" * 60)
    
    root_dir = "DDI"
    images_dir = os.path.join(root_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    num_images = 150
    data = []
    
    # Fitzpatrick skin tone groups: 12 (light), 34 (medium), 56 (dark)
    skin_tones = [12, 34, 56]
    
    np.random.seed(42) # For reproducible mock data
    
    for i in range(1, num_images + 1):
        filename = f"{i:06d}.png"
        filepath = os.path.join(images_dir, filename)
        
        # Generate a random skin-like background color
        # Light skins have higher RGB values, dark skins have lower/warmer RGB values
        skin_tone = int(np.random.choice(skin_tones))
        if skin_tone == 12:
            bg_color = (np.random.randint(220, 255), np.random.randint(180, 220), np.random.randint(160, 200))
        elif skin_tone == 34:
            bg_color = (np.random.randint(180, 220), np.random.randint(140, 180), np.random.randint(110, 150))
        else: # 56
            bg_color = (np.random.randint(100, 140), np.random.randint(70, 100), np.random.randint(50, 80))
            
        img = Image.new("RGB", (299, 299), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Determine if it's malignant (1) or benign (0)
        is_malignant = int(np.random.choice([0, 1]))
        
        # Add a random "lesion" (a circle/ellipse in the center)
        # Malignant lesions: darker, larger, irregular
        # Benign lesions: more uniform, symmetric
        center_x, center_y = 150, 150
        if is_malignant:
            radius_x = np.random.randint(30, 60)
            radius_y = np.random.randint(30, 60)
            lesion_color = (np.random.randint(30, 70), np.random.randint(20, 50), np.random.randint(10, 40))
        else:
            radius_x = np.random.randint(20, 40)
            radius_y = radius_x # symmetric
            lesion_color = (np.random.randint(70, 110), np.random.randint(50, 80), np.random.randint(35, 60))
            
        draw.ellipse(
            [center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y],
            fill=lesion_color
        )
        
        img.save(filepath)
        
        data.append({
            "DDI_file": filename,
            "malignant": is_malignant,
            "skin_tone": skin_tone
        })
        
        if i % 30 == 0:
            print(f"Generated {i}/{num_images} mock images...")
            
    # Save metadata CSV
    csv_path = os.path.join(root_dir, "ddi_metadata.csv")
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    print("=" * 60)
    print(f"Mock DDI dataset successfully generated at '{root_dir}'!")
    print(f"Total images: {num_images}")
    print(f"Metadata file: {csv_path}")
    print("=" * 60)

if __name__ == "__main__":
    generate_mock_dataset()
