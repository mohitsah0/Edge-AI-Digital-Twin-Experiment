"""
Training Script for Edge-AI 1D-CNN Cyberattack Detector
Paper: Edge-AI-Driven Digital Twin for Real-Time Cyberattack Detection
       and Resilient Control in Renewable Microgrids
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from cnn_1d import EdgeAI_CNN


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Edge-AI 1D-CNN")
    p.add_argument("--data",       default="../data/microgrid_dataset.npz")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--output",     default="edge_ai_cnn.pth")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ─── Training loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += X.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += X.size(0)
    return total_loss / total, correct / total


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cpu")

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    data   = np.load(args.data)
    X, y   = data["X"], data["y"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=args.seed)
    X_val, X_test, y_val, y_test   = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=args.seed)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    def make_loader(Xd, yd, shuffle=True):
        ds = TensorDataset(torch.FloatTensor(Xd), torch.LongTensor(yd))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    model     = EdgeAI_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True)

    best_val_acc, t0 = 0.0, time.time()
    log = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        log.append({"epoch": epoch, "train_acc": tr_acc, "val_acc": vl_acc})

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), args.output)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"train_loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
                  f"val_loss={vl_loss:.4f} acc={vl_acc*100:.2f}%")

    train_time = time.time() - t0
    print(f"\nBest val acc : {best_val_acc*100:.2f}%")
    print(f"Train time   : {train_time:.1f}s")

    # Final test evaluation
    model.load_state_dict(torch.load(args.output))
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc*100:.2f}%")

    # Save training log
    Path("training_log.json").write_text(json.dumps(log, indent=2))
    print("Training log saved to training_log.json")


if __name__ == "__main__":
    main()
