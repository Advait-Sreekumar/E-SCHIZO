import matplotlib.pyplot as plt
import numpy as np
import pyautogui
import os

# CONFIGURATION
FILE_PATH = "screen_position.txt"
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size() # Automatically detect screen size

def load_gaze_data(filepath):
    """Reads x,y coordinates from the text file."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return [], []

    xs, ys = [], []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    x, y = int(parts[0]), int(parts[1])
                    xs.append(x)
                    ys.append(y)
            except ValueError:
                continue # Skip bad lines
    return xs, ys

def create_visualization():
    print(f"Reading data from {FILE_PATH}...")
    x, y = load_gaze_data(FILE_PATH)

    if not x:
        print("No data found! Run the tracker and calibrate first.")
        return

    # --- PLOT SETUP ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set plot limits to match your screen resolution
    ax.set_xlim(0, SCREEN_WIDTH)
    ax.set_ylim(SCREEN_HEIGHT, 0) # Invert Y axis so (0,0) is top-left (like a screen)
    
    # Set background color to represent a screen
    ax.set_facecolor('black')
    
    # --- 1. HEATMAP GENERATION ---
    # We use a 2D histogram to calculate density
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[64, 48], range=[[0, SCREEN_WIDTH], [0, SCREEN_HEIGHT]])
    
    # Display the heatmap (using 'inferno' colormap for fire-like effect)
    # alpha=0.6 makes it semi-transparent so you can see the lines on top
    ax.imshow(heatmap.T, origin='upper', extent=[0, SCREEN_WIDTH, SCREEN_HEIGHT, 0], cmap='inferno', alpha=0.6, interpolation='gaussian')

    # --- 2. SCANPATH (TRAJECTORY) ---
    # Draw the path of eye movement
    ax.plot(x, y, color='cyan', linewidth=1, alpha=0.5, label='Eye Path')

    # Mark Start (Green) and End (Red)
    ax.scatter(x[0], y[0], color='lime', s=100, label='Start', edgecolors='white', zorder=5)
    ax.scatter(x[-1], y[-1], color='red', s=100, label='End', edgecolors='white', zorder=5)

    # --- FINAL DETAILS ---
    ax.set_title(f"Eye Tracking Session Analysis\n({len(x)} data points)", color='white')
    ax.set_xlabel("Screen Width (px)")
    ax.set_ylabel("Screen Height (px)")
    ax.legend()
    plt.tight_layout()
    
    # Save the result
    plt.savefig("gaze_heatmap.png", dpi=100)
    print("Visualization saved as 'gaze_heatmap.png'")
    
    # Show the interactive window
    plt.show()

if __name__ == "__main__":
    create_visualization()