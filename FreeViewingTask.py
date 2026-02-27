import pygame
import pyautogui
import pandas as pd
import os
import time
import sys

# --- CONFIGURATION ---
STIMULI_FOLDER = "stimuli"  # Folder containing images
VIEWING_TIME = 5.0          # Seconds per image
FIXATION_TIME = 1.0         # Seconds for crosshair between images
OUTPUT_FILE = "session_data.csv"

def get_image_files(folder):
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    return [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]

def run_experiment():
    # 1. Setup Pygame (Full Screen)
    pygame.init()
    screen_info = pygame.display.Info()
    SCREEN_W, SCREEN_H = screen_info.current_w, screen_info.current_h
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False) # Hide cursor so it doesn't distract

    # 2. Load Stimuli
    if not os.path.exists(STIMULI_FOLDER):
        print(f"Error: Folder '{STIMULI_FOLDER}' not found. Please create it and add images.")
        pygame.quit()
        return

    images = get_image_files(STIMULI_FOLDER)
    if not images:
        print("No images found in stimuli folder.")
        pygame.quit()
        return

    # 3. Data Storage
    # Columns: Timestamp, Image Name, Gaze X, Gaze Y
    gaze_data = []

    font = pygame.font.SysFont('Arial', 50)
    clock = pygame.time.Clock()

    print("Starting Experiment. Press ESC to abort.")

    # --- MAIN LOOP ---
    for img_name in images:
        # A. Show Fixation Cross
        screen.fill((128, 128, 128)) # Grey background
        pygame.draw.line(screen, (0,0,0), (SCREEN_W//2 - 20, SCREEN_H//2), (SCREEN_W//2 + 20, SCREEN_H//2), 5)
        pygame.draw.line(screen, (0,0,0), (SCREEN_W//2, SCREEN_H//2 - 20), (SCREEN_W//2, SCREEN_H//2 + 20), 5)
        pygame.display.flip()
        
        # Wait for fixation time
        start_fix = time.time()
        while time.time() - start_fix < FIXATION_TIME:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

        # B. Show Stimulus
        img_path = os.path.join(STIMULI_FOLDER, img_name)
        img_surface = pygame.image.load(img_path)
        img_surface = pygame.transform.scale(img_surface, (SCREEN_W, SCREEN_H))
        
        screen.blit(img_surface, (0, 0))
        pygame.display.flip()

        # C. Record Data
        start_view = time.time()
        while time.time() - start_view < VIEWING_TIME:
            # Check for quit
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    # Save partial data
                    pd.DataFrame(gaze_data).to_csv(OUTPUT_FILE, index=False)
                    pygame.quit()
                    return

            # --- CAPTURE GAZE (MOUSE POSITION) ---
            # MonitorTracking.py moves the mouse, so we just read the mouse pos.
            mx, my = pyautogui.position()
            
            gaze_data.append({
                "timestamp": time.time(),
                "image": img_name,
                "gaze_x": mx,
                "gaze_y": my
            })

            # Record at ~60Hz
            clock.tick(60)

    # 4. Save Data
    df = pd.DataFrame(gaze_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Session Complete. Data saved to {OUTPUT_FILE}")
    
    pygame.quit()

if __name__ == "__main__":
    run_experiment()