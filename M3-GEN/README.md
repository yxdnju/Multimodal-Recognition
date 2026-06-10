📖 Overview

Robust Cross-Modal Alignment Network(M3-GEN) is a deep learning framework designed for mortality prediction and disease diagnosis in missing modality scenarios. It effectively integrates multiple physiological signals (HR, ABP, Resp) with clinical text notes and demographic features, maintaining robust performance even when some modalities are partially or completely missing.

Key Features

🩺 Multi-View Signal Encoding: Independent encoding with mask awareness + adaptive temporal window GRU

🔄 Cross-Modal Alignment: Contrastive learning to align semantic representations across modalities

🧬 Variational Inference: Uncertainty quantification via VAE-based reparameterization

🎯 Multi-Task Learning: Main classifier + auxiliary classifiers for modality-specific predictions

📊 Attention Visualization: Comprehensive tools for interpreting cross-modal interactions


🚀 Quick Start

Installation

# Clone repository
git clone https://github.com/yourusername/RCMAN.git
cd RCMAN

# Create conda environment
conda create -n rcman python=3.8
conda activate rcman

# Install dependencies
pip install torch>=1.9.0 transformers numpy scikit-learn matplotlib seaborn networkx
