import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import datetime

# --- CONFIGURATION ---
MODEL_PATH = "best_msnet_stabilized.pth"
# Threshold found during training (0.50 - 0.53)
DECISION_THRESHOLD = 0.50 

# --- MODEL ARCHITECTURE (Must match training) ---
class SFB_Stage(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels // 2
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv_reduce = nn.Conv1d(in_channels, self.out_channels, 1)
        self.norm2 = nn.LayerNorm(self.out_channels)
        self.query = nn.Conv1d(self.out_channels, self.out_channels, 1)
        self.key   = nn.Conv1d(self.out_channels, self.out_channels, 1)
        self.value = nn.Conv1d(self.out_channels, self.out_channels, 1)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x_in = x.transpose(1, 2)
        x_norm = self.norm1(x_in.transpose(1, 2)).transpose(1, 2)
        x_reduced = self.conv_reduce(x_norm) 
        x_norm2 = self.norm2(x_reduced.transpose(1, 2)).transpose(1, 2)
        Q = self.query(x_norm2); K = self.key(x_norm2); V = self.value(x_norm2)
        scores = torch.bmm(Q.transpose(1, 2), K) / (self.out_channels ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.bmm(attn_weights, V.transpose(1, 2)).transpose(1, 2)
        output = self.dropout(attn_output) + x_reduced
        return output.transpose(1, 2)

class CCB_Module(nn.Module):
    def __init__(self, current_channels):
        super().__init__()
        self.threshold_learner = nn.Linear(100, 1) 
        self.relu = nn.ReLU()
        self.update_conv = nn.Conv1d(current_channels, current_channels // 2, 1)
    def forward(self, stimulus_features, current_center, density_vector):
        diff = stimulus_features - current_center 
        d_min = density_vector.min(dim=1, keepdim=True)[0]
        d_max = density_vector.max(dim=1, keepdim=True)[0]
        d_norm = (density_vector - d_min) / (d_max - d_min + 1e-6)
        epsilon = self.threshold_learner(d_norm)
        d_thresholded = self.relu(d_norm - epsilon)
        d_weights = d_thresholded / (d_thresholded.sum(dim=1, keepdim=True) + 1e-6)
        mean_shift_vector = (diff * d_weights.unsqueeze(2)).sum(dim=1, keepdim=True)
        new_center = self.update_conv((current_center + mean_shift_vector).transpose(1, 2)).transpose(1, 2)
        return new_center

class MSNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 2048 Input (ResNet), 512 Internal (Stabilized)
        self.embedding = nn.Conv2d(2048, 512, 1) 
        self.sfb_stages = nn.ModuleList([
            SFB_Stage(512), SFB_Stage(256), SFB_Stage(128), SFB_Stage(64)
        ])
        self.ccg_conv = nn.Conv1d(512, 256, 1)
        self.ccb_modules = nn.ModuleList([
            CCB_Module(256), CCB_Module(128), CCB_Module(64)
        ])
        self.final_threshold = nn.Linear(100, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(32), nn.Dropout(0.5), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) 
        x_embed = self.embedding(x) 
        f_0 = x_embed.mean(dim=3).transpose(1, 2)
        f_norm = F.normalize(f_0, p=2, dim=2)
        density = torch.bmm(f_norm, f_norm.transpose(1, 2)).sum(dim=2)
        center_idx = torch.argmax(density, dim=1)
        batch_indices = torch.arange(f_0.size(0)).to(f_0.device)
        center_raw = f_0[batch_indices, center_idx].unsqueeze(1)
        current_center = self.ccg_conv(center_raw.transpose(1, 2)).transpose(1, 2)
        current_stimulus_features = f_0
        for i in range(3):
            next_stimulus_features = self.sfb_stages[i](current_stimulus_features)
            next_center = self.ccb_modules[i](next_stimulus_features, current_center, density)
            current_stimulus_features = next_stimulus_features; current_center = next_center
        final_stimulus = self.sfb_stages[3](current_stimulus_features)
        d_min = density.min(dim=1, keepdim=True)[0]; d_max = density.max(dim=1, keepdim=True)[0]
        d_norm = (density - d_min) / (d_max - d_min + 1e-6)
        d_w = (F.relu(d_norm - self.final_threshold(d_norm)) / (F.relu(d_norm - self.final_threshold(d_norm)).sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(2)
        diff = final_stimulus - current_center
        shift = (diff * d_w).sum(dim=1, keepdim=True)
        final_center = current_center + shift
        return self.classifier(final_center.squeeze(1))

def generate_report(patient_file):
    print("\n--- EMS DIAGNOSTIC SYSTEM ---")
    
    # 1. Load Data
    if not os.path.exists(patient_file):
        print(f"❌ Error: File '{patient_file}' not found.")
        return
    
    try:
        data = np.load(patient_file) # Shape [100, 14, 2048]
        print(f"✅ Patient Data Loaded. Shape: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model '{MODEL_PATH}' not found. Please download it from Drive.")
        return

    device = torch.device("cpu") # Safe for local inference
    model = MSNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ AI Model Loaded Successfully.")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        return

    # 3. Predict
    with torch.no_grad():
        # Add batch dimension [1, 100, 14, 2048]
        tensor_data = torch.FloatTensor(data).unsqueeze(0).to(device)
        probability = model(tensor_data).item()

    # 4. Generate Report
    diagnosis = "SCHIZOPHRENIA DETECTED" if probability >= DECISION_THRESHOLD else "HEALTHY CONTROL"
    confidence = probability if probability > 0.5 else (1.0 - probability)
    
    report = f"""
    ===================================================
                CLINICAL DIAGNOSTIC REPORT
    ===================================================
    Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Patient File: {os.path.basename(patient_file)}
    ---------------------------------------------------
    
    ANALYSIS RESULT:
    ----------------
    Diagnosis:  [{diagnosis}]
    Confidence: {confidence:.2%}
    Risk Score: {probability:.4f} (Threshold: {DECISION_THRESHOLD})
    
    ---------------------------------------------------
    INTERPRETATION:
    The MSNet Deep Learning model analyzed the patient's 
    eye movement scanpaths and fixation features.
    
    - Scores near 0.0 indicate Healthy.
    - Scores near 1.0 indicate Schizophrenia.
    
    This tool is an assistive device and should be verified
    by a clinical professional.
    ===================================================
    """
    
    print(report)
    
    # Save to text file
    with open("Final_Report.txt", "w") as f:
        f.write(report)
    print("📄 Report saved to 'Final_Report.txt'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python DiagnosticTool.py <path_to_patient_npy_file>")
        print("Example: python DiagnosticTool.py patient_session.npy")
    else:
        generate_report(sys.argv[1])