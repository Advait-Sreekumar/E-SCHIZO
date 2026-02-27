import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2

# CONFIGURATION
DATA_FILE = "session_data.csv"
STIMULI_FOLDER = "stimuli"
OUTPUT_FOLDER = "analysis_results"

def analyze_session():
    if not os.path.exists(DATA_FILE):
        print("No session data found. Run the experiment first.")
        return
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Load Data
    df = pd.read_csv(DATA_FILE)
    
    # Get unique images presented
    presented_images = df['image'].unique()

    for img_name in presented_images:
        print(f"Analyzing {img_name}...")
        
        # 1. Filter data for this specific image
        img_data = df[df['image'] == img_name]
        x = img_data['gaze_x'].values
        y = img_data['gaze_y'].values
        
        # 2. Load the original image background
        img_path = os.path.join(STIMULI_FOLDER, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Original image {img_name} missing.")
            continue
            
        # Read image (OpenCV reads BGR, convert to RGB)
        bg_img = cv2.imread(img_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        h, w, _ = bg_img.shape

        # 3. Create Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Show background image
        ax.imshow(bg_img, extent=[0, w, h, 0])

        # A. Draw Scanpath (Blue Line)
        ax.plot(x, y, color='cyan', linewidth=2, alpha=0.6, label='Scanpath')
        
        # B. Draw Start/End points
        if len(x) > 0:
            ax.scatter(x[0], y[0], c='lime', s=100, edgecolors='black', label='Start', zorder=5)
            ax.scatter(x[-1], y[-1], c='red', s=100, edgecolors='black', label='End', zorder=5)

        # C. Generate Heatmap
        # We clamp data to image dimensions to avoid errors
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        
        # Create heatmap histogram
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[64, 48], range=[[0, w], [0, h]])
        
        # Overlay heatmap (using alpha for transparency)
        # 'jet' or 'inferno' are good colormaps. alpha=0.4 lets the image show through.
        ax.imshow(heatmap.T, origin='upper', extent=[0, w, h, 0], cmap='jet', alpha=0.4, interpolation='gaussian')

        ax.set_title(f"Gaze Analysis: {img_name}")
        ax.axis('off') # Hide axes for cleaner look
        ax.legend(loc='upper right')

        # Save
        save_path = os.path.join(OUTPUT_FOLDER, f"analysis_{img_name}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    print(f"Analysis complete! Check the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    analyze_session()