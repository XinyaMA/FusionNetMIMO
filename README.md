# FusionNetMIMO
A demo for ML solution for signal detection.....

16-QAM Dataset Generation: Automatically generate synthetic datasets using Rayleigh fading channels and AWGN noise.

Deep Learning Models:

FCN (Fully Connected Network) with dropout and Adam optimizer.

CNN with convolutional feature extraction and OneCycle learning rate scheduling.

FusionNet, a dual-branch architecture combining CNN and FCN with focal loss.

Classical Detectors: ML, ZF, MMSE, SIC, and SDR (using cvxpy for convex optimization).

Performance Comparison:

Training and validation accuracy/loss visualization.

BER vs. SNR evaluation (0–20 dB).

Benchmark against theoretical BER.

# Requirements
Programming Language: Python 3.10

Deep Learning: TensorFlow 2.15, PyTorch 2.5

Optimization: CVXPY

Data Handling: NumPy, HDF5 (h5py)

Visualization: Matplotlib

Environment: Anaconda (Spyder / Colab), CUDA 12.6 + cuDNN 8.9

# Files structure
├── comp_CNN_FCN_FusionNet.py      # Main pipeline: dataset generation, FCN, CNN, FusionNet, BER comparison
├── mxythesis_comparison.py        # Classical detection algorithms (ML, ZF, MMSE, SIC, SDR)
├── MNIST测试.py                    # Model testing using MNIST dataset for structure validation
├── thesis.pdf                     # Full research paper including methodology and results
└── dataset_16qam.h5               # Generated dataset (created at runtime)









