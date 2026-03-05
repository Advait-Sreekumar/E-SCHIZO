import cv2
import numpy as np
import time
import random
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

# --- CONFIGURATION ---
GAZE_FILE = "screen_position.txt"
WIDTH, HEIGHT = 1920, 1080 
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
MAX_TRIALS = 20  
TARGET_OFFSETS = [400, 600] 
X_TOLERANCE = 150 

class SchizoEyeResearch:
    def __init__(self):
        self.state = "IDLE"
        self.results = []
        self.all_gaze_x = [] 
        self.fixation_wobble = [] 
        self.timer = 0
        self.target_x = 0
        self.final_stats = {}

        try:
            self.model = joblib.load('schizo_model.pkl')
            self.scaler = joblib.load('schizo_scaler.pkl')
            self.model_loaded = True
            print("✅ Models Loaded")
        except:
            self.model_loaded = False
            print("⚠️ Models missing. Using raw metrics.")

    def get_live_gaze_x(self):
        try:
            with open(GAZE_FILE, "r") as f:
                lines = f.readlines()
                if lines: 
                    gx = int(lines[-1].strip().split(',')[0])
                    if self.state in ["STIMULUS", "FIXATION"]:
                        self.all_gaze_x.append(gx)
                    return gx
        except: return CENTER_X
        return CENTER_X

    def save_unified_report(self):
        """Combines text metrics and the waveform into a single PNG image"""
        if len(self.all_gaze_x) < 2: return
        
        # Calculate velocity
        velocity = np.diff(self.all_gaze_x)
        
        # Create a figure with two subplots: Top for Text, Bottom for Waveform
        fig, (ax_text, ax_wave) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
        fig.patch.set_facecolor('#0a0a0a')

        # 1. Plot the Text Report
        ax_text.axis('off')
        report_str = "SCHIZOEYE DIAGNOSTIC SUMMARY\n" + "="*30 + "\n\n"
        for k, v in self.final_stats.items():
            report_str += f"{k}: {v}\n"
        report_str += f"\nGenerated: {time.ctime()}"
        
        ax_text.text(0.1, 0.5, report_str, transform=ax_text.transAxes, 
                     color='white', fontsize=14, family='monospace', verticalalignment='center')

        # 2. Plot the Waveform
        ax_wave.set_facecolor('#111111')
        ax_wave.plot(velocity, color='#00FF00', linewidth=1)
        ax_wave.set_title("Saccade Velocity Profile", color='white', fontsize=12)
        ax_wave.set_ylabel("Velocity (px/sample)", color='gray')
        ax_wave.tick_params(colors='gray')
        for spine in ax_wave.spines.values():
            spine.set_color('#333333')

        plt.tight_layout()
        plt.savefig("diagnostic_summary.png", facecolor=fig.get_facecolor())
        plt.close()
        print("📊 Unified report saved to diagnostic_summary.png")

    def calculate_metrics(self):
        if not self.results: return
        rts = [r['rt'] * 1000 for r in self.results] 
        m_rt = np.mean(rts)
        cv = np.std(rts) / m_rt if m_rt > 0 else 0
        err_count = len([r for r in self.results if not r['correct']])
        
        risk = 0.0
        if self.model_loaded:
            eff = m_rt / max((1.0 - (err_count/MAX_TRIALS)), 0.01)
            feat = np.array([[np.clip(m_rt, 200, 1200), np.clip(cv, 0, 0.6), m_rt*0.7, skew(rts), eff]])
            risk = round(self.model.predict_proba(self.scaler.transform(feat))[0][1] * 100, 2)

        wobble = np.std(self.fixation_wobble) if self.fixation_wobble else 0
        reliability = 100 - (cv * 50) - (wobble / 5)
        
        self.final_stats = {
            "Risk Score": f"{risk}%",
            "Classification": "SCHIZOPHRENIA" if risk >= 38 else "CONTROL",
            "Reliability Index": f"{max(0, min(100, reliability)):.1f}%",
            "Behavioral Accuracy": f"{(MAX_TRIALS-err_count)/MAX_TRIALS*100:.0f}%",
            "Mean RT": f"{m_rt:.1f}ms",
            "CV (Stability)": f"{cv:.3f}"
        }
        self.save_unified_report()

    def run(self):
        cv2.namedWindow("SchizoEye Research", cv2.WND_PROP_FULLSCREEN)
        while True:
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            gaze_x = self.get_live_gaze_x()
            
            # Header
            cv2.rectangle(canvas, (0,0), (WIDTH, 70), (20, 20, 20), -1)
            trial_text = f"TRIAL: {len(self.results) + 1} / {MAX_TRIALS}" if self.state != "FINISHED" else "DONE"
            cv2.putText(canvas, trial_text, (CENTER_X - 100, 45), 1, 1.8, (255, 255, 255), 2)

            # Gaze Ball
            dist_x = abs(gaze_x - CENTER_X)
            ball_color = (0, 255, 0) if dist_x < X_TOLERANCE else (0, 0, 255)
            if self.state != "FINISHED":
                cv2.circle(canvas, (gaze_x, CENTER_Y), 12, ball_color, -1)

            if self.state == "IDLE":
                cv2.putText(canvas, "SPACE TO START", (CENTER_X-150, CENTER_Y), 1, 1.5, (255,255,255), 2)

            elif self.state == "FIXATION":
                cv2.circle(canvas, (CENTER_X, CENTER_Y), 25, (40, 40, 40), -1)
                if dist_x < X_TOLERANCE:
                    self.fixation_wobble.append(dist_x)
                    if self.timer == 0: self.timer = time.time()
                    if time.time() - self.timer > 1.2:
                        self.state, self.target_x = "STIMULUS", CENTER_X + (random.choice([-1, 1]) * random.choice(TARGET_OFFSETS))
                        self.timer = time.time()
                else: self.timer = 0

            elif self.state == "STIMULUS":
                cv2.circle(canvas, (self.target_x, CENTER_Y), 50, (0, 0, 200), -1)
                offset = gaze_x - CENTER_X
                if abs(offset) > 250:
                    rt = time.time() - self.timer
                    corr = (self.target_x > CENTER_X and offset < 0) or (self.target_x < CENTER_X and offset > 0)
                    self.results.append({'rt': rt, 'correct': corr})
                    if len(self.results) >= MAX_TRIALS:
                        self.calculate_metrics()
                        self.state = "FINISHED"
                    else: self.state, self.timer = "COOLDOWN", time.time()

            elif self.state == "COOLDOWN" and time.time() - self.timer > 0.8:
                self.state, self.timer = "FIXATION", 0

            elif self.state == "FINISHED":
                cv2.rectangle(canvas, (CENTER_X-450, CENTER_Y-250), (CENTER_X+450, CENTER_Y+200), (10, 10, 10), -1)
                y = CENTER_Y - 180
                for k, v in self.final_stats.items():
                    c = (0,0,255) if "SCHIZ" in str(v) or "Risk" in k else (255,255,255)
                    cv2.putText(canvas, f"{k}: {v}", (CENTER_X-400, y), 1, 1.8, c, 2)
                    y += 60
                cv2.putText(canvas, "UNIFIED REPORT OVERWRITTEN", (CENTER_X-200, CENTER_Y+170), 1, 1, (0, 255, 0), 1)

            cv2.imshow("SchizoEye Research", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and self.state == "IDLE": self.state = "FIXATION"
            if key == ord('q'): break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SchizoEyeResearch().run()