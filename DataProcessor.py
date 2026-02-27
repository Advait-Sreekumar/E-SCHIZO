import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import pyautogui
from scipy.signal import savgol_filter

# --- CONFIGURATION ---
CSV_FILE = "session_data.csv"
STIMULI_FOLDER = "stimuli"
OUTPUT_FILE = "patient_data.npy"
SEQUENCE_LENGTH = 100 

def process_data():
    print("--- STARTING DATA PROCESSING ---")
    
    # Get the actual screen resolution the experiment was run on
    try:
        SCREEN_W, SCREEN_H = pyautogui.size()
        print(f"Detected Screen Resolution: {SCREEN_W}x{SCREEN_H}")
    except:
        SCREEN_W, SCREEN_H = 1920, 1080 # Fallback
        
    print("Loading ResNet50...")
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    resnet = models.resnet50(weights=weights)
    resnet = nn.Sequential(*list(resnet.children())[:-1]) 
    resnet.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found. Run the experiment first!")
        return

    df = pd.read_csv(CSV_FILE)
    
    # 1. Apply Clinical-Grade Smoothing Filter
    window_length = min(31, len(df) // 2 * 2 + 1) 
    if window_length > 3:
        df['gaze_x'] = savgol_filter(df['gaze_x'], window_length, 3)
        df['gaze_y'] = savgol_filter(df['gaze_y'], window_length, 3)
        df['gaze_x'] = df['gaze_x'].clip(lower=0, upper=SCREEN_W)
        df['gaze_y'] = df['gaze_y'].clip(lower=0, upper=SCREEN_H)

    # 2. Extract Features
    features_list = []
    indices = np.linspace(0, len(df)-1, SEQUENCE_LENGTH, dtype=int)
    print(f"Extracting features from {len(indices)} smoothed fixation points...")
    
    with torch.no_grad():
        for i in indices:
            row = df.iloc[i]
            img_name = row['image']
            gaze_x = int(row['gaze_x'])
            gaze_y = int(row['gaze_y'])
            
            img_path = os.path.join(STIMULI_FOLDER, img_name)
            
            try:
                image = Image.open(img_path).convert('RGB')
                
                # 🌟 THE FIX: Stretch image to match the fullscreen Pygame experiment
                image = image.resize((SCREEN_W, SCREEN_H), Image.Resampling.LANCZOS)
                w, h = image.size
                
                # Crop a patch around the gaze point
                patch_size = 224
                left = max(0, min(gaze_x - patch_size//2, w - patch_size))
                top = max(0, min(gaze_y - patch_size//2, h - patch_size))
                patch = image.crop((left, top, left + patch_size, top + patch_size))
                
                input_tensor = preprocess(patch).unsqueeze(0)
                feature_vector = resnet(input_tensor).squeeze() 
                features_list.append(feature_vector.numpy())
                
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                features_list.append(np.zeros(2048))

    raw_features = np.stack(features_list)
    final_data = np.tile(raw_features[:, np.newaxis, :], (1, 14, 1))
    
    print(f"✅ Processing Complete.")
    print(f"Output Shape: {final_data.shape}")
    
    np.save(OUTPUT_FILE, final_data)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()