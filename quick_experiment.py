"""
Edge-AI Digital Twin - Quick Experiment (10 epochs)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import json

np.random.seed(42)
torch.manual_seed(42)

print("Starting experiment...")

# Generate smaller dataset
def generate_data(n=10000):
    t = np.linspace(0, 1000, n)
    data = np.zeros((n, 32))
    labels = np.zeros(n, dtype=int)
    
    spc = n // 6
    
    # Base signals
    v = 1.0 + 0.02 * np.sin(2 * np.pi * 0.1 * t)
    f = 60.0 + 0.05 * np.cos(2 * np.pi * 0.05 * t)
    p = 0.8 + 0.1 * np.sin(2 * np.pi * 0.08 * t)
    
    # Class 0: Normal
    data[0:spc, :6] = np.column_stack([v[0:spc], f[0:spc], p[0:spc], p[0:spc]*0.3, p[0:spc]/v[0:spc], np.arctan2(p[0:spc]*0.3, p[0:spc])])
    data[0:spc, :6] += np.random.normal(0, 0.001, (spc, 6))
    
    # Class 1: FDI
    data[spc:2*spc, :6] = np.column_stack([v[spc:2*spc]+0.05, f[spc:2*spc]-0.15, p[spc:2*spc], p[spc:2*spc]*0.3, p[spc:2*spc]/v[spc:2*spc], np.arctan2(p[spc:2*spc]*0.3, p[spc:2*spc])])
    data[spc:2*spc, :6] += np.random.normal(0, 0.001, (spc, 6))
    labels[spc:2*spc] = 1
    
    # Class 2: DoS
    data[2*spc:3*spc, :6] = np.column_stack([v[2*spc-5:3*spc-5], f[2*spc-5:3*spc-5], p[2*spc:3*spc], p[2*spc:3*spc]*0.3, p[2*spc:3*spc]/v[2*spc:3*spc], np.arctan2(p[2*spc:3*spc]*0.3, p[2*spc:3*spc])])
    data[2*spc:3*spc, :6] += np.random.normal(0, 0.005, (spc, 6))
    labels[2*spc:3*spc] = 2
    
    # Class 3: GPS Spoofing
    data[3*spc:4*spc, :6] = np.column_stack([v[3*spc:4*spc], f[3*spc:4*spc], p[3*spc-10:4*spc-10], p[3*spc-10:4*spc-10]*0.3, p[3*spc:4*spc]/v[3*spc:4*spc], np.arctan2(p[3*spc:4*spc]*0.3, p[3*spc:4*spc])+0.3])
    data[3*spc:4*spc, :6] += np.random.normal(0, 0.001, (spc, 6))
    labels[3*spc:4*spc] = 3
    
    # Class 4: Replay
    ridx = np.random.randint(0, spc, spc)
    data[4*spc:5*spc, :6] = np.column_stack([v[ridx], f[ridx], p[ridx], p[ridx]*0.3, p[ridx]/v[ridx], np.arctan2(p[ridx]*0.3, p[ridx])])
    data[4*spc:5*spc, :6] += np.random.normal(0, 0.0005, (spc, 6))
    labels[4*spc:5*spc] = 4
    
    # Class 5: Measurement Manipulation
    data[5*spc:6*spc, :6] = np.column_stack([v[5*spc:6*spc]*1.03, f[5*spc:6*spc]*0.998, p[5*spc:6*spc]*1.05, p[5*spc:6*spc]*0.285, p[5*spc:6*spc]/v[5*spc:6*spc]*1.04, np.arctan2(p[5*spc:6*spc]*0.3, p[5*spc:6*spc])])
    data[5*spc:6*spc, :6] += np.random.normal(0, 0.001, (spc, 6))
    labels[5*spc:6*spc] = 5
    
    # Fill remaining features
    for i in range(6, 32):
        data[:, i] = data[:, i % 6] * (1 + 0.1 * np.random.randn(n))
    
    return data, labels

print("Generating dataset...")
X, y = generate_data(10000)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=128, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=128, shuffle=False)

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(256 * 4, 128)
        self.fc2 = nn.Linear(128, 6)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# Train
print("Training...")
train_start = time.time()
for epoch in range(10):
    model.train()
    for bX, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bX), by)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for bX, by in val_loader:
            _, pred = torch.max(model(bX), 1)
            val_total += by.size(0)
            val_correct += (pred == by).sum().item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/10 - Val Acc: {100*val_correct/val_total:.2f}%")

train_time = time.time() - train_start

# Test
print("Testing...")
model.eval()
all_preds, all_labels, latencies = [], [], []

with torch.no_grad():
    for bX, by in test_loader:
        t0 = time.time()
        out = model(bX)
        latencies.append((time.time() - t0) * 1000 / bX.size(0))
        _, pred = torch.max(out, 1)
        all_preds.extend(pred.numpy())
        all_labels.extend(by.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
acc = accuracy_score(all_labels, all_preds) * 100
prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
cm = confusion_matrix(all_labels, all_preds)

tn = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)
fp = np.sum(cm, axis=0) - np.diag(cm)
far = np.mean(fp / (fp + tn + 1e-10)) * 100

mean_lat = np.mean(latencies)
p95_lat = np.percentile(latencies, 95)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Accuracy:     {acc:.2f}%")
print(f"Precision:    {prec:.2f}%")
print(f"Recall:       {rec:.2f}%")
print(f"F1-Score:     {f1:.2f}%")
print(f"FAR:          {far:.2f}%")
print(f"Mean Latency: {mean_lat:.4f} ms")
print(f"P95 Latency:  {p95_lat:.4f} ms")
print(f"Train Time:   {train_time:.2f} sec")
print(f"Parameters:   {total_params:,}")

# Per-class
class_names = ['Normal', 'FDI', 'DoS', 'GPS', 'Replay', 'Manip']
print("\nPer-Class Accuracy:")
for i in range(6):
    mask = (all_labels == i)
    if mask.sum() > 0:
        cacc = accuracy_score(all_labels[mask], all_preds[mask]) * 100
        print(f"  {class_names[i]:<10} {cacc:>6.2f}%")

# Save results
results = {
    'accuracy': float(acc),
    'precision': float(prec),
    'recall': float(rec),
    'f1_score': float(f1),
    'far': float(far),
    'mean_latency_ms': float(mean_lat),
    'p95_latency_ms': float(p95_lat),
    'train_time_sec': float(train_time),
    'parameters': int(total_params),
    'confusion_matrix': cm.tolist()
}

with open('/home/mohitsah0/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results.json")
