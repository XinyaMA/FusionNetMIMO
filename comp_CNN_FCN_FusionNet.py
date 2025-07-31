# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 22:42:32 2025

@author: MA
"""

# 16-QAM dataset generator
import numpy as np
import h5py

# ========== parameters ==========
Nt, Nr = 2, 2       # Antenna Configuration
M = 16              # modulation order
snr_dB = 15         # signal-to-noise ratio(SNR)
num_samples = 100000
save_path = 'dataset_16qam.h5'  # Save in HDF5 format

# 16-QAM constellation diagram (power normalised)
constellation = np.array([(x + 1j*y)/np.sqrt(10)
                         for x in [-3, -1, 1, 3]
                         for y in [-3, -1, 1, 3]])

# ========== Generate data ==========
X = np.zeros((num_samples, 2*Nt*Nr + 2*Nr), dtype=np.float32)
Y = np.zeros(num_samples, dtype=np.int32)

for i in range(num_samples):
    # Generating launch symbols
    tx_sym = np.random.randint(0, M, size=Nt)
    tx_sig = constellation[tx_sym]

    # Generate channel matrix
    H = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)

    # generate noise
    noise_power = 10 ** (-snr_dB / 10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(Nr) + 1j*np.random.randn(Nr))

    # reception
    rx_sig = H @ tx_sig + noise

    # === feature extraction ===
    H_real = H.real.flatten() / np.sqrt(2)   # Flatten to 1D array
    H_imag = H.imag.flatten() / np.sqrt(2)
    rx_real = rx_sig.real / np.sqrt(10)
    rx_imag = rx_sig.imag / np.sqrt(10)

    # Splicing features
    X[i] = np.concatenate([H_real, H_imag, rx_real, rx_imag])

    # label encoding
    Y[i] = tx_sym[0] * M + tx_sym[1]

# ========== save ==========
# split training set and validation set
split = int(0.9 * num_samples)
with h5py.File(save_path, 'w') as f:
    f.create_dataset('X_train', data=X[:split])
    f.create_dataset('Y_train', data=Y[:split])
    f.create_dataset('X_val', data=X[split:])
    f.create_dataset('Y_val', data=Y[split:])

print(f"The dataset has been saved to: {save_path}")

# ========== data validation ==========
# Check the dimensions
print("\nData dimension validation:")
print(f"X_train shape: {X[:split].shape} (should be ({split}, 12))")
print(f"Y_train shape: {Y[:split].shape}")

# Check a sample
idx = np.random.randint(num_samples)
print(f"\nsamples {idx} detail:")
print("input:", X[idx])
print("transmitted symbols:", tx_sym)
print("label encoding:", Y[idx])



"""第二部分

用自己的数据集训练FCN模型2*2MIMO

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import h5py  # For loading .h5 datasets

# 1. Loading custom 16-QAM datasets
def load_16qam_data(file_path):
    with h5py.File(file_path, 'r') as f:
        x_train = f['X_train'][:]
        y_train = f['Y_train'][:]
        x_val = f['X_val'][:]
        y_val = f['Y_val'][:]

    # Converting labels to one-hot encoding (class 256)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=256)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=256)

    return (x_train, y_train), (x_val, y_val)

# Load data (replace with my dataset path)
(x_train, y_train), (x_val, y_val) = load_16qam_data('dataset_16qam.h5')

# 2. Data validation
print("\n=== Data dimension validation ===")
print(f"Training set input dimensions: {x_train.shape} (should be (80000, 12))")
print(f"Training set labelling dimensions: {y_train.shape} (should be (80000, 256))")
print(f"Validation set input dimensions: {x_val.shape} (should be (20000, 12))")

# 3. Constructing an adapted FCN model
model = Sequential([
    Dense(1024, activation='relu', input_shape=(12,)),  # Input layer adapted to 12-dimensional features
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),   # An intermediate layer that matches the dimensions of the output layer
    Dense(256, activation='softmax') # Output Layer Adaptation 256 Class Classification
])

# 4. compilation model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Configuring Callback Functions
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10,
                 restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=5, min_lr=1e-6)
]

# 6. model training
print("\n=== Start training ===")
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=256,  # Increase batch size to accommodate larger datasets
                    validation_data=(x_val, y_val),
                    callbacks=callbacks,
                    verbose=2)

# 7. Performance Visualisation
plt.figure(figsize=(12, 5))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('FCN (16-QAM-2*2MIMO) Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.grid(True)

# loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.title('FCN (16-QAM-2*2MIMO) Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 8. Final evaluation
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"\nFinal validation accuracy: {val_acc*100:.2f}%")

model_fcn = model  # Save my FCN model

model.save('fcn_model.h5')
print("Model saved successfully.")


"""第三部分

用自己的数据集测试的CNN模型

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt


# 在PyTorch代码前释放TensorFlow资源
import tensorflow as tf
tf.keras.backend.clear_session()

torch.cuda.empty_cache()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Dataset Class with proper reshaping
class QAMDataset(Dataset):
    def __init__(self, file_path, is_train=True):
        with h5py.File(file_path, 'r') as f:
            if is_train:
                self.x = f['X_train'][:]
                self.y = f['Y_train'][:]
            else:
                self.x = f['X_val'][:]
                self.y = f['Y_val'][:]

        # Reshape to 4-channel 2x2 image
        reshaped_x = []
        for sample in self.x:
            h_real = sample[:4].reshape(2, 2)
            h_imag = sample[4:8].reshape(2, 2)
            rx_real = np.zeros((2, 2))
            rx_real[:, 0] = sample[8:10]
            rx_imag = np.zeros((2, 2))
            rx_imag[:, 0] = sample[10:12]
            combined = np.stack([h_real, h_imag, rx_real, rx_imag], axis=0)
            reshaped_x.append(combined)

        self.x = np.array(reshaped_x, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# Enhanced CNN Model
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=4, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        return self.features(x)

# Initialize model
model = EnhancedCNN().to(device)
print(model)

# Data loaders
train_dataset = QAMDataset("dataset_16qam.h5", is_train=True)
test_dataset = QAMDataset("dataset_16qam.h5", is_train=False)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, pin_memory=True)

# Training configuration
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.05,
                                        steps_per_epoch=len(train_loader),
                                        epochs=10)

def train_model():
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    model.train()
    for epoch in range(10):
        epoch_train_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

        # Calculate training metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_correct += predicted.eq(target).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100. * val_correct / len(test_dataset)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

    # Plotting
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='training')
    plt.plot(val_accuracies, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('CNN (16-QAM-2*2MIMO) Accuracy Curve')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='training')
    plt.plot(val_losses, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN (16-QAM-2*2MIMO) Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Final evaluation
    model.eval()
    final_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            final_correct += predicted.eq(target).sum().item()

    final_acc = 100. * final_correct / len(test_dataset)
    print(f"\nFinal Validation Accuracy: {final_acc:.2f}%")

# Start training
train_model()

model_cnn = model  # Save my CNN model

torch.save(model_cnn.state_dict(), 'cnn_model.pth')  # 保存模型参数




"第四部分：用自己的数据集测试FusionNet"


import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
tf.keras.backend.clear_session()  # 清除TensorFlow占用的资源
import tensorflow_addons as tfa

import torch
torch.cuda.empty_cache()

# ========== Data Loading & Preprocessing ==========
def load_data(path):
    with h5py.File(path, 'r') as f:
        return (
            np.array(f['X_train']),
            np.array(f['Y_train']),
            np.array(f['X_val']),
            np.array(f['Y_val'])
        )

X_train, Y_train, X_val, Y_val = load_data('dataset_16qam.h5')

# Reshape for CNN input (4x3x1)
X_train_cnn = X_train.reshape(-1, 4, 3, 1)
X_val_cnn = X_val.reshape(-1, 4, 3, 1)

# Keep original features for FCN
X_train_fcn = X_train.copy()
X_val_fcn = X_val.copy()

# Convert labels to one-hot (256 classes)
Y_train = keras.utils.to_categorical(Y_train, 256)
Y_val = keras.utils.to_categorical(Y_val, 256)

# ========== Enhanced FusionNet Architecture ==========
def build_fusionnet():
    # CNN Branch
    cnn_input = layers.Input(shape=(4, 3, 1)) 
    x = layers.Conv2D(64, (4,4), activation='gelu', padding='same')(cnn_input)
    x = layers.BatchNormalization()(x)  
    x = layers.Conv2D(128, (3,3), activation='gelu', padding='same')(x)
    x = layers.BatchNormalization()(x)   
    x = layers.Conv2D(256, (2,2), activation='gelu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (2,2), activation='gelu', padding='same')(x)
    x = layers.BatchNormalization()(x) 
    x = layers.Conv2D(1024, (2,2), activation='gelu', padding='same')(x)
    x = layers.BatchNormalization()(x)   
    x = layers.GlobalMaxPooling2D()(x)

    
    # FCN Branch
    fcn_input = layers.Input(shape=(12,))
    y = layers.Dense(1024)(fcn_input)
    y = layers.LeakyReLU(alpha=0.1)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(512)(fcn_input)
    y = layers.LeakyReLU(alpha=0.1)(y)
    y = layers.BatchNormalization()(y)

    # Feature Fusion
    combined = layers.concatenate([x, y])
    
    # Deep Processing
    z = layers.Dense(1024, activation='relu')(combined)
    z = layers.Dense(512, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.1)(z)
    
    # Output Layer
    outputs = layers.Dense(256, activation='softmax')(z)
    
    return keras.Model(inputs=[cnn_input, fcn_input], outputs=outputs)

# ========== Learning Rate Schedule ==========
def lr_schedule(epoch):
    return 1e-3 * (0.8 ** epoch)

# ========== Model Compilation ==========
model = build_fusionnet()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=tfa.losses.SigmoidFocalCrossEntropy(),
    metrics=['accuracy']
)

# ========== Training Configuration ==========
history = model.fit(
    [X_train_cnn, X_train_fcn], Y_train,
    batch_size=256,
    epochs=10,
    validation_data=([X_val_cnn, X_val_fcn], Y_val),
    callbacks=[LearningRateScheduler(lr_schedule)],
    verbose=1
)

# ========== Evaluation ==========
val_loss, val_acc = model.evaluate([X_val_cnn, X_val_fcn], Y_val, verbose=0)
print(f"\nValidation Accuracy: {val_acc*100:.2f}%")

# ========== Visualization ==========
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('FusionNet (16-QAM-2*2MIMO) Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('FusionNet (16-QAM-2*2MIMO) Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ========== Model Export ==========
model.save('fusionnet_final.h5')
print("Model saved successfully.")

torch.cuda.empty_cache()  # 释放PyTorch的GPU缓存


"第五部分：验证FCN和CNN和FusionNet的BER-SNR图像"


import torch
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.keras.backend.clear_session()  # 清除TensorFlow占用的资源

import torch
torch.cuda.empty_cache()


# ========== Parameters ==========
Nt, Nr = 2, 2
M = 16
bits_per_symbol = int(np.log2(M))
constellation = np.array([(x + 1j*y)/np.sqrt(10) for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]])
SNR_range = np.arange(0, 21, 2)
num_samples_per_snr = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load FCN and FusionNet models ==========
model_fusion = load_model('fusionnet_final.h5')
model_fcn = load_model('fcn_model.h5')

# ========== CNN model ==========
class EnhancedCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(4, 128, kernel_size=4, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256)
        )

    def forward(self, x):
        return self.features(x)



# ========== Load CNN model ==========
model_cnn = EnhancedCNN().to(device)
try:
    model_cnn.load_state_dict(torch.load('cnn_model.pth', map_location=device))
except RuntimeError as e:
    print(f"Failed to load the model, please check if the model structure matches: {str(e)}")
    exit()

model_cnn.eval()

# ========== Optimized bit conversion function ==========
def symbols_to_bits(symbols):
    """Convert symbol-indexed arrays to bitstreams (batch processing supported)"""
    symbols = np.asarray(symbols).flatten()
    bits = np.unpackbits(symbols.astype(np.uint8).reshape(-1, 8)[:, -4:])  # 取后4位
    return bits.flatten()

# ========== BER calculation function ==========
def calculate_ber(model_type, snr_db):
    noise_power = 10 ** (-snr_db / 10)
    total_errors = 0

    for _ in range(num_samples_per_snr // 1000):
        # === Signal Generation ===
        tx_sym = np.random.randint(0, M, (1000, Nt))
        tx_sig = constellation[tx_sym]
        H = (np.random.randn(1000, Nr, Nt) + 1j*np.random.randn(1000, Nr, Nt)) / np.sqrt(2)
        noise = np.sqrt(noise_power/2) * (np.random.randn(1000, Nr) + 1j*np.random.randn(1000, Nr))
        rx_sig = H @ tx_sig[..., None] + noise[..., None]
        rx_sig = rx_sig.squeeze()

        # === Feature extraction ===
        H_real = H.real.reshape(1000, 4)/np.sqrt(2)
        H_imag = H.imag.reshape(1000, 4)/np.sqrt(2)
        rx_real = rx_sig.real.reshape(1000, 2)/np.sqrt(10)
        rx_imag = rx_sig.imag.reshape(1000, 2)/np.sqrt(10)

        # === Model prediction ===
        if model_type == 'FCN':
            X_input = np.concatenate([H_real, H_imag, rx_real, rx_imag], axis=1)
            pred = np.argmax(model_fcn.predict(X_input, verbose=0), axis=1)
        
        elif model_type == 'CNN':
            # Build inputs that match the structure of the training data (4 channels 2x2)
            h_real = H_real.reshape(1000, 2, 2)
            h_imag = H_imag.reshape(1000, 2, 2)
            
            # Correctly populate rx signals to 2x2
            rx_real_pad = np.zeros((1000, 2, 2))
            rx_real_pad[:, :, 0] = rx_real
            rx_imag_pad = np.zeros((1000, 2, 2))
            rx_imag_pad[:, :, 0] = rx_imag
            
            cnn_input = np.stack([h_real, h_imag, rx_real_pad, rx_imag_pad], axis=1)
            tensor_input = torch.tensor(cnn_input, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = model_cnn(tensor_input).argmax(dim=1).cpu().numpy()
        
        elif model_type == 'FusionNet':
            # Correct input shape (4x3x1)
            X_fcn = np.concatenate([H_real, H_imag, rx_real, rx_imag], axis=1)
            X_cnn = X_fcn.reshape(-1, 4, 3, 1)  # correct shape
            pred = np.argmax(model_fusion.predict([X_cnn, X_fcn], verbose=0), axis=1)

        # === Ber calculation ===
        tx_bits = symbols_to_bits(tx_sym)
        pred_syms = np.column_stack([pred // M, pred % M])
        rx_bits = symbols_to_bits(pred_syms)
        total_errors += np.sum(tx_bits != rx_bits)

    return total_errors / (num_samples_per_snr * Nt * bits_per_symbol)

# ========== Main loop ==========
ber_results = {'FCN': [], 'CNN': [], 'FusionNet': []}
for snr in SNR_range:
    print(f"\nProcessing SNR = {snr} dB...")
    for model_name in ber_results:
        ber = calculate_ber(model_name, snr)
        ber_results[model_name].append(ber)
        print(f"{model_name}: BER = {ber:.2e}")

# ========== visualization ==========
plt.figure(figsize=(10,6))
markers = {'FCN': 's', 'CNN': '^', 'FusionNet': 'o'}
for model_name in ber_results:
    plt.semilogy(SNR_range, ber_results[model_name], 
                marker=markers[model_name], 
                label=f'{model_name}')

# # Theoratical curve
# snr_linear = 10**(SNR_range/10)
# ber_theo = 3/8 * (1 - np.sqrt(3/(3 + snr_linear)))
# plt.semilogy(SNR_range, ber_theo, 'k--', label='Theory')

plt.title('2x2 MIMO 16-QAM BER Performance(FCN vs CNN vs FusionNet)')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate(BER)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.xticks(SNR_range)
plt.ylim(2e-2, 2e-1)
plt.tight_layout()
plt.show()


# "第六部分SER"

# # ========== SER 计算函数 ==========
# def calculate_ser(model_type, snr_db):
#     noise_power = 10 ** (-snr_db / 10)
#     total_symbol_errors = 0

#     for _ in range(num_samples_per_snr // 1000):
#         # === 信号生成 ===
#         tx_sym = np.random.randint(0, M, (1000, Nt))
#         tx_sig = constellation[tx_sym]
#         H = (np.random.randn(1000, Nr, Nt) + 1j*np.random.randn(1000, Nr, Nt)) / np.sqrt(2)
#         noise = np.sqrt(noise_power/2) * (np.random.randn(1000, Nr) + 1j*np.random.randn(1000, Nr))
#         rx_sig = H @ tx_sig[..., None] + noise[..., None]
#         rx_sig = rx_sig.squeeze()

#         # === 特征提取 ===
#         H_real = H.real.reshape(1000, 4)/np.sqrt(2)
#         H_imag = H.imag.reshape(1000, 4)/np.sqrt(2)
#         rx_real = rx_sig.real.reshape(1000, 2)/np.sqrt(10)
#         rx_imag = rx_sig.imag.reshape(1000, 2)/np.sqrt(10)

#         # === 模型推理 ===
#         if model_type == 'FCN':
#             X_input = np.concatenate([H_real, H_imag, rx_real, rx_imag], axis=1)
#             pred = np.argmax(model_fcn.predict(X_input, verbose=0), axis=1)

#         elif model_type == 'CNN':
#             h_real = H_real.reshape(1000, 2, 2)
#             h_imag = H_imag.reshape(1000, 2, 2)
#             rx_real_pad = np.zeros((1000, 2, 2))
#             rx_real_pad[:, :, 0] = rx_real
#             rx_imag_pad = np.zeros((1000, 2, 2))
#             rx_imag_pad[:, :, 0] = rx_imag
#             cnn_input = np.stack([h_real, h_imag, rx_real_pad, rx_imag_pad], axis=1)
#             tensor_input = torch.tensor(cnn_input, dtype=torch.float32).to(device)
#             with torch.no_grad():
#                 pred = model_cnn(tensor_input).argmax(dim=1).cpu().numpy()

#         elif model_type == 'FusionNet':
#             X_fcn = np.concatenate([H_real, H_imag, rx_real, rx_imag], axis=1)
#             X_cnn = X_fcn.reshape(-1, 4, 3, 1)
#             pred = np.argmax(model_fusion.predict([X_cnn, X_fcn], verbose=0), axis=1)

#         # === SER计算 ===
#         pred_syms = np.column_stack([pred // M, pred % M])
#         total_symbol_errors += np.sum(np.any(pred_syms != tx_sym, axis=1))

#     return total_symbol_errors / (num_samples_per_snr)

# # ========== 主循环计算 SER ==========
# ser_results = {'FCN': [], 'CNN': [], 'FusionNet': []}
# for snr in SNR_range:
#     print(f"SER计算中 SNR = {snr} dB...")
#     for model_name in ser_results:
#         ser = calculate_ser(model_name, snr)
#         ser_results[model_name].append(ser)
#         print(f"{model_name}: SER = {ser:.2e}")

# # ========== SER 可视化 ==========
# plt.figure(figsize=(10,6))
# for model_name in ser_results:
#     plt.semilogy(SNR_range, ser_results[model_name], 
#                  marker=markers[model_name], 
#                  linestyle='--', label=f'{model_name} SER')
# plt.title('2x2 MIMO 16-QAM SER Performance (FCN vs CNN vs FusionNet)')
# plt.xlabel('SNR (dB)')
# plt.ylabel('Symbol Error Rate (SER)')
# plt.grid(True, which='both', linestyle='--')
# plt.legend()
# plt.xticks(SNR_range)
# plt.tight_layout()
# plt.show()

