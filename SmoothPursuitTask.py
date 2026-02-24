import pygame
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt

# ================= CONFIGURATION =================
INPUT_FILE = "screen_position.txt"
TASK_DURATION = 15          # 15 seconds of actual movement
COUNTDOWN_TIME = 3          # 3 second text countdown
FIXATION_TIME = 2           # 2 seconds stationary dot before movement
FPS = 60
DOT_RADIUS = 35
PURSUIT_FREQUENCY = 0.25    # Clinical standard frequency
# =================================================

def tail_file(filename):
    try:
        with open(filename, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
            return last_line.strip() if last_line else None
    except:
        return None

def show_countdown(screen, font):
    clock = pygame.time.Clock()
    for i in range(COUNTDOWN_TIME, 0, -1):
        start_tick = time.perf_counter()
        while time.perf_counter() - start_tick < 1:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
            screen.fill((0, 0, 0))
            text = font.render(f"Starting in {i}", True, (255, 255, 255))
            rect = text.get_rect(center=(screen.get_width()//2, screen.get_height()//2))
            screen.blit(text, rect)
            pygame.display.flip()
            clock.tick(FPS)
    return True

def run_smooth_pursuit_task():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    W, H = screen.get_size()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 60)

    if not show_countdown(screen, font):
        pygame.quit()
        return pd.DataFrame()

    data_log = []
    amplitude = (W // 2) - (DOT_RADIUS * 3)
    omega = 2 * np.pi * PURSUIT_FREQUENCY
    phase_shift = -np.pi / 2  # Start at left edge
    
    start_time = time.perf_counter()
    running = True

    while running:
        current_time = time.perf_counter() - start_time
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Logic for stationary start vs movement
        if current_time < FIXATION_TIME:
            target_x = (W // 2) + amplitude * np.sin(phase_shift)
            is_moving = False
        elif current_time < (FIXATION_TIME + TASK_DURATION):
            pursuit_time = current_time - FIXATION_TIME
            target_x = (W // 2) + amplitude * np.sin(omega * pursuit_time + phase_shift)
            is_moving = True
        else:
            running = False
            continue

        target_y = H // 2

        # Data collection
        if is_moving:
            line = tail_file(INPUT_FILE)
            if line:
                try:
                    parts = line.split(',')
                    data_log.append({
                        'time': current_time - FIXATION_TIME,
                        'target_x': target_x,
                        'eye_x': int(float(parts[0]))
                    })
                except: pass

        # UI Rendering
        screen.fill((0, 0, 0))
        pygame.draw.line(screen, (60, 60, 60), (0, H//2), (W, H//2), 1)
        pygame.draw.circle(screen, (255, 0, 0), (int(target_x), int(target_y)), DOT_RADIUS)
            
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    return pd.DataFrame(data_log)

def analyze_data(df):
    if df.empty or len(df) < 100:
        print("Error: No data captured.")
        return

    # 1. Jitter Removal
    df['eye_x_clean'] = medfilt(df['eye_x'], kernel_size=15) 
    mean_dt = df['time'].diff().mean() or 0.016

    # 2. Velocity Calculation
    df['v_eye_fast'] = savgol_filter(df['eye_x_clean'], 15, 2, deriv=1, delta=mean_dt)
    df['v_eye_smooth'] = savgol_filter(df['eye_x_clean'], 31, 3, deriv=1, delta=mean_dt)
    df['v_target'] = savgol_filter(df['target_x'], 11, 2, deriv=1, delta=mean_dt)

    # 3. Saccade Detection (Lowered to 2000 to find the middle ground)
    saccade_threshold = 2000 
    df['is_saccade_raw'] = abs(df['v_eye_fast']) > saccade_threshold
    
    # Apply "Debounce": A saccade must be separated by at least 100ms to be counted as a new one
    # This prevents counting webcam noise spikes as multiple clinical events
    df['saccade_start'] = (df['is_saccade_raw'] & ~df['is_saccade_raw'].shift(1).fillna(False))
    
    # Filter saccades that happen too close together (within 0.1 seconds)
    last_saccade_time = -1.0
    valid_saccades = 0
    df['is_saccade'] = False # Reset for clean visualization

    for i, row in df.iterrows():
        if row['saccade_start']:
            if row['time'] - last_saccade_time > 0.1:
                valid_saccades += 1
                last_saccade_time = row['time']
                df.at[i, 'is_saccade'] = True # Mark for the graph

    saccade_freq = valid_saccades / TASK_DURATION

    # 4. Gain Calculation
    pursuit_mask = (abs(df['v_eye_fast']) < saccade_threshold) & (abs(df['v_target']) > 150)
    if pursuit_mask.any():
        df['gain_raw'] = abs(df['v_eye_smooth'] / df['v_target'])
        avg_gain = df.loc[pursuit_mask, 'gain_raw'].clip(0, 1.5).mean()
    else:
        avg_gain = 0.0

    # 5. Risk Calculation (Clinical Logic)
    # Penalize low gain (<0.8) and high saccade frequency (>2.0 Hz)
    gain_penalty = max(0, (0.85 - avg_gain) * 100)
    saccade_penalty = max(0, (saccade_freq - 1.5) * 20)
    final_risk = min(99, gain_penalty + saccade_penalty)

    # 6. Clinical Results Printing
    print(f"\n--- FINAL CLINICAL ANALYSIS ---")
    print(f"Velocity Gain:     {avg_gain:.3f} (Normal: >0.85)")
    print(f"Saccade Frequency: {saccade_freq:.2f} Hz (Normal: <1.5 Hz)")
    print(f"Total Saccades:    {valid_saccades}")
    print(f"Risk Probability:  {final_risk:.1f}%")

    # 7. GRAPHING
    plt.figure(figsize=(12, 8), facecolor='white')
    
    # Subplot 1: Position
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['time'], df['target_x'], 'r-', label='Target Path', linewidth=2)
    ax1.plot(df['time'], df['eye_x_clean'], 'b-', alpha=0.5, label='Filtered Eye Path')
    
    # Highlight the valid saccades we counted
    saccade_points = df[df['is_saccade']]
    if not saccade_points.empty:
        ax1.scatter(saccade_points['time'], saccade_points['eye_x_clean'], color='orange', s=50, label='Detected Saccade', zorder=5)

    ax1.set_title(f"Position Tracking (Risk: {final_risk:.1f}%)")
    ax1.set_ylabel("Screen Pixels (X)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Subplot 2: Velocity
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df['time'], abs(df['v_target']), 'r--', alpha=0.4, label='Target Velocity')
    ax2.plot(df['time'], abs(df['v_eye_smooth']), 'b-', label='Smoothed Eye Velocity')
    ax2.axhline(saccade_threshold, color='orange', linestyle=':', label='Detection Threshold')
    
    ax2.set_title(f"Velocity Gain Analysis (Avg: {avg_gain:.2f})")
    ax2.set_ylabel("Velocity (px/s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0, max(2500, saccade_threshold + 500))
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = run_smooth_pursuit_task()
    analyze_data(data)