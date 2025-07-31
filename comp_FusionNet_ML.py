# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 21:39:14 2025

@author: MA
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import torch

# ========== 参数设置 ==========
Nt, Nr = 2, 2
M = 16  # 统一为16-QAM
bits_per_symbol = int(np.log2(M))
SNR_range = np.arange(0, 21, 2)
num_samples_per_snr = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载FusionNet模型 ==========
model_fusion = load_model('fusionnet_final.h5')

# ========== ML检测算法BER计算函数 ==========
def calculate_ml_ber(snr_db):
    noise_power = 10 ** (-snr_db / 10)
    total_errors = 0
    constellation = np.array([(x + 1j*y)/np.sqrt(10) for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]])

    for _ in range(num_samples_per_snr // 1000):
        # 生成信号
        tx_sym = np.random.randint(0, M, (1000, Nt))
        tx_sig = constellation[tx_sym]
        H = (np.random.randn(1000, Nr, Nt) + 1j*np.random.randn(1000, Nr, Nt)) / np.sqrt(2)
        noise = np.sqrt(noise_power/2) * (np.random.randn(1000, Nr) + 1j*np.random.randn(1000, Nr))
        rx_sig = H @ tx_sig[..., None] + noise[..., None]
        rx_sig = rx_sig.squeeze()

        # ML检测
        min_dist = np.inf * np.ones(1000)
        best_sym = np.zeros((1000, Nt), dtype=int)
        for s1 in range(M):
            for s2 in range(M):5
                test_sig = np.array([constellation[s1], constellation[s2]])
                dist = np.linalg.norm(rx_sig - H @ test_sig, axis=1)**2
                mask = dist < min_dist
                best_sym[mask] = [s1, s2]
                min_dist[mask] = dist[mask]

        # BER计算
        tx_bits = symbols_to_bits(tx_sym)
        rx_bits = symbols_to_bits(best_sym)
        total_errors += np.sum(tx_bits != rx_bits)

    return total_errors / (num_samples_per_snr * Nt * bits_per_symbol)

# ========== FusionNet BER计算函数 ==========
def calculate_fusion_ber(snr_db):
    noise_power = 10 ** (-snr_db / 10)
    total_errors = 0
    constellation = np.array([(x + 1j*y)/np.sqrt(10) for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]])

    for _ in range(num_samples_per_snr // 1000):
        # 生成信号
        tx_sym = np.random.randint(0, M, (1000, Nt))
        tx_sig = constellation[tx_sym]
        H = (np.random.randn(1000, Nr, Nt) + 1j*np.random.randn(1000, Nr, Nt)) / np.sqrt(2)
        noise = np.sqrt(noise_power/2) * (np.random.randn(1000, Nr) + 1j*np.random.randn(1000, Nr))
        rx_sig = H @ tx_sig[..., None] + noise[..., None]
        rx_sig = rx_sig.squeeze()

        # 特征提取
        H_real = H.real.reshape(1000, 4)/np.sqrt(2)
        H_imag = H.imag.reshape(1000, 4)/np.sqrt(2)
        rx_real = rx_sig.real.reshape(1000, 2)/np.sqrt(10)
        rx_imag = rx_sig.imag.reshape(1000, 2)/np.sqrt(10)
        X_fcn = np.concatenate([H_real, H_imag, rx_real, rx_imag], axis=1)
        X_cnn = X_fcn.reshape(-1, 4, 3, 1)  # 匹配FusionNet输入形状

        # 预测
        pred = np.argmax(model_fusion.predict([X_cnn, X_fcn], verbose=0), axis=1)
        pred_syms = np.column_stack([pred // M, pred % M])

        # BER计算
        tx_bits = symbols_to_bits(tx_sym)
        rx_bits = symbols_to_bits(pred_syms)
        total_errors += np.sum(tx_bits != rx_bits)

    return total_errors / (num_samples_per_snr * Nt * bits_per_symbol)

# ========== 通用工具函数 ==========
def symbols_to_bits(symbols):
    symbols = np.asarray(symbols).flatten()
    return np.unpackbits(symbols.astype(np.uint8)).reshape(-1, 8)[:, -4:].flatten()

# ========== 主计算循环 ==========
ber_ml = []
ber_fusion = []

for snr in SNR_range:
    print(f"\nProcessing SNR = {snr} dB...")
    ber_ml.append(calculate_ml_ber(snr))
    ber_fusion.append(calculate_fusion_ber(snr))
    print(f"ML: {ber_ml[-1]:.2e} | FusionNet: {ber_fusion[-1]:.2e}")

# ========== 理论曲线计算 ==========
snr_linear = 10**(SNR_range/10)
ber_theo = 3/8 * (1 - np.sqrt(3/(3 + snr_linear)))  # 16-QAM理论近似

# ========== 绘图 ==========
plt.figure(figsize=(10,6))
plt.semilogy(SNR_range, ber_ml, '-o', label='ML Detection')
plt.semilogy(SNR_range, ber_fusion, '-s', label='FusionNet')
plt.semilogy(SNR_range, ber_theo, 'k--', label='Theoretical 16-QAM')

plt.title('2x2 MIMO 16-QAM BER Performance Comparison')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.xticks(SNR_range)
plt.ylim(1e-3, 1e0)
plt.tight_layout()
plt.show()