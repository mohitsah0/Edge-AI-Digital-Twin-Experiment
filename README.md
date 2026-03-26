# Edge-AI-Digital-Twin-Experiment
This repository contains the complete source code, experimental setup, and reproducibility package
Quick Start
1. Installation
# Clone repository
git clone [https://github.com/yourusername/edge-ai-digital-twin.git](https://github.com/mohitsah0/Edge-AI-Digital-Twin-Experiment/)
cd edge-ai-digital-twin

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
2. Generate Dataset
# Generate 50,000 synthetic microgrid samples (6 classes)
python data/generate_dataset.py --samples 50000 --output data/microgrid_dataset.npz
3. Train Edge-AI Detection Model
# Train 1D-CNN model
python src/models/train.py \\
    --data data/microgrid_dataset.npz \\
    --epochs 50 \\
    --batch_size 256 \\
    --lr 0.001 \\
    --output models/edge_ai_cnn.pth
4. Run Full Experiments
# Run all experiments (detection + resilient control)
python experiments/run_experiments.py \\
    --model models/edge_ai_cnn.pth \\
    --output results/
5. Reproduce Paper Results
# Generate all figures and tables from paper
python experiments/generate_paper_results.py \\
    --results results/experimental_results.json \\
    --output paper/figures/
