# FusionNetMIMO
A demo of a convolutional neural network solution for signal detection.....

- 16-QAM Dataset Generation: Automatically generate synthetic datasets using Rayleigh fading channels and AWGN noise.

- Deep Learning Models:

FCN (Fully Connected Network) with dropout and Adam optimizer.

CNN with convolutional feature extraction and OneCycle learning rate scheduling.

FusionNet, a dual-branch architecture combining CNN and FCN with focal loss.

- Classical Detectors: ML, ZF, MMSE, SIC, and SDR (using cvxpy for convex optimization).

- Performance Comparison:

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

```bash
├── comp_CNN_FCN_FusionNet.py      
├── comparison.m                
├── MNIST测试.py                    
├── comp_FusionNet_ML.py    
└── dataset_16qam.h5     
```
- ** comp_CNN_FCN_FusionNet.py **: Main pipeline, dataset generation, FCN, CNN, FusionNet, BER comparison
- ** comparison.m **: Classical detection algorithms (ML, ZF, MMSE, SIC, SDR) by MATLAB
- ** MNIST测试.py **: Model testing using MNIST dataset for structure validation
- ** comp_FusionNet_ML.py **: Comparison between FusionNet detection and ML detection methods
- ** dataset_16qam.h5 **: Generated dataset (created at runtime)



