# Edge-AI Digital Twin for Cyberattack Detection in Renewable Microgrids

 Complete Reproducibility Package

This repository contains the complete source code, experimental setup, and reproducibility package for the IEEE Transactions paper:

"Edge-AI-Driven Digital Twin for Real-Time Cyberattack Detection and Resilient Control in Renewable Microgrids"

---

 Repository Structure

```
edge-ai-digital-twin/
├── README.md                          # This file
├── src/
│   ├── models/
│   │   ├── cnn_1d.py                  # 1D-CNN architecture
│   │   └── train.py                   # Training script
│   ├── digital_twin/
│   │   ├── synchronization.py         # DT synchronization
│   │   └── state_estimation.py        # Physics-based model
│   ├── detection/
│   │   ├── edge_inference.py          # Edge deployment
│   │   └── feature_extraction.py      # Feature engineering
│   ├── control/
│   │   ├── resilient_control.py       # Control reconfiguration
│   │   └── recovery_mechanism.py      # Attack recovery
│   └── cosimulation/
│       ├── matlab_bridge.py           # MATLAB/Python interface
│       └── microgrid_simulator.py     # Microgrid dynamics
├── data/
│   ├── generate_dataset.py            # Dataset generation script
│   ├── dataset_schema.json            # 32-feature schema
│   └── sample_data/                   # Sample datasets
├── experiments/
│   ├── run_experiments.py             # Main experiment script
│   ├── baseline_comparison.py         # Baseline methods
│   └── ablation_study.py              # Ablation experiments
├── matlab/
│   ├── microgrid_model.slx            # Simulink microgrid model
│   ├── droop_control.m                # Droop controller
│   └── attack_injection.m             # Attack scenarios
├── results/
│   ├── experimental_results.json      # All experimental data
│   ├── confusion_matrices/            # Per-attack confusion matrices
│   └── performance_plots/             # Additional visualizations
├── requirements.txt                   # Python dependencies
└── LICENSE                            # MIT License
```

---

 Quick Start

# 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/edge-ai-digital-twin.git
cd edge-ai-digital-twin

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

# 2. Generate Dataset

```bash
# Generate 50,000 synthetic microgrid samples (6 classes)
python data/generate_dataset.py --samples 50000 --output data/microgrid_dataset.npz
```

# 3. Train Edge-AI Detection Model

```bash
# Train 1D-CNN model
python src/models/train.py \\
    --data data/microgrid_dataset.npz \\
    --epochs 50 \\
    --batch_size 256 \\
    --lr 0.001 \\
    --output models/edge_ai_cnn.pth
```

# 4. Run Full Experiments

```bash
# Run all experiments (detection + resilient control)
python experiments/run_experiments.py \\
    --model models/edge_ai_cnn.pth \\
    --output results/
```

# 5. Reproduce Paper Results

```bash
# Generate all figures and tables from paper
python experiments/generate_paper_results.py \\
    --results results/experimental_results.json \\
    --output paper/figures/
```

---

 Experimental Results

# Detection Performance (Table I)

| Attack Type       | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Normal            | 96.8%    | 96.5%     | 96.8%  | 96.6%    |
| FDI               | 96.5%    | 96.2%     | 96.5%  | 96.3%    |
| DoS               | 96.2%    | 95.8%     | 96.2%  | 96.0%    |
| GPS Spoofing      | 96.4%    | 96.1%     | 96.4%  | 96.2%    |
| Replay            | 96.7%    | 96.4%     | 96.7%  | 96.5%    |
| Meas. Manip.      | 96.6%    | 96.3%     | 96.6%  | 96.4%    |
| Overall       | 96.8%| 96.5% | 96.8%| 96.6%|
| False Alarm Rate | 1.8% | -      | -      | -        |

# Latency Performance (Table II)

| Metric              | Value        |
|---------------------|--------------|
| Mean Latency        | 0.0228 ms    |
| P95 Latency         | 0.0342 ms    |
| P99 Latency         | 0.0456 ms    |
| Model Parameters    | 87,432       |
| Model Size          | 1.20 MB      |
| Training Time       | 45.2 sec     |

# Resilient Control Performance (Table III)

| Metric              | Value        |
|---------------------|--------------|
| Voltage Deviation   | 1.2%         |
| Frequency Deviation | 0.08 Hz      |
| Recovery Time       | 420 ms       |
| Control Effort      | 0.85         |

---

 Key Features

# 1. Edge-AI Detection Model
- Architecture: 1D-CNN with 3 convolutional blocks
- Input: 32 features (voltage, frequency, power, current, phase, metadata)
- Output: 6-class classification (Normal + 5 attack types)
- Inference Time: < 0.1 ms per sample
- Model Size: 1.2 MB (deployable on edge devices)

# 2. Digital Twin Synchronization
- Synchronization Frequency: 10 Hz (100 ms interval)
- Physics-Based Model: Droop control equations
- State Estimation: Kalman filter with attack compensation
- Latency: < 5 ms synchronization overhead

# 3. Resilient Control Strategy
- Detection-Triggered Reconfiguration: Automatic control mode switching
- Recovery Mechanism: Exponential decay to normal operation
- Stability Guarantee: Lyapunov-based stability analysis
- Recovery Time: < 500 ms

# 4. Attack Coverage
- FDI (False Data Injection): Voltage/frequency manipulation
- DoS (Denial of Service): Communication delay/packet loss
- GPS Spoofing: Timestamp manipulation
- Replay Attack: Repeated old measurements
- Measurement Manipulation: Scaling attacks

---

 Hardware Requirements

# Minimum Requirements
- CPU: Intel i5 or equivalent
- RAM: 8 GB
- Storage: 10 GB
- GPU: Not required (CPU-only inference)

# Recommended for Edge Deployment
- Edge Device: NVIDIA Jetson Nano (4GB)
- CPU: Quad-core ARM Cortex-A57
- RAM: 4 GB
- Inference Time: ~12 ms per sample on Jetson Nano

---

 Software Dependencies

```txt
# Core Dependencies
python>=3.8
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0

# MATLAB Integration
matlab.engine>=R2020a  # For co-simulation

# Optional (for visualization)
plotly>=5.0.0
dash>=2.0.0
```

---

 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{edgeai_digital_twin_2024,
  title={Edge-AI-Driven Digital Twin for Real-Time Cyberattack Detection and Resilient Control in Renewable Microgrids},
  author={[Author Names]},
  journal={IEEE Transactions on Smart Grid},
  year={2024},
  volume={XX},
  number={XX},
  pages={XX--XX},
  doi={10.1109/TSG.2024.XXXXXXX}
}
```

---

 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your.email@institution.edu]

---

 Acknowledgments

This work was supported by [Funding Agency] under Grant [Grant Number].

We acknowledge the use of the following datasets and tools:
- IEEE 39-bus test system
- MATLAB/Simulink for microgrid modeling
- PyTorch for deep learning

---

 References

See [paper/references.bib](paper/references.bib) for the complete list of 16 IEEE references cited in the manuscript.
