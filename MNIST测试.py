# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 20:13:13 2025

@author: MA
"""

"""
第一部分： FCN的MNIST数据集测试
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Data preprocessing
# Normalise and flatten the image
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Tags converted to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. Data validation
print("\n=== Data dimension validation ===")
print(f"Training set input dimensions: {x_train.shape} (should be (60000, 784))")
print(f"Training set labelling dimensions: {y_train.shape} (should be (60000, 10))")
print(f"Test set input dimensions: {x_test.shape} (should be (10000, 784))")

# 4. Constructing FCN models (adapting MNIST)
model = Sequential([
    Dense(1024, activation='relu', input_shape=(784,)),  # Input layer adjusted to 784 dimensions
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')  # Output layer changed to 10 nodes
])

# 5. compilation model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Configuring Callback Functions
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                 restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=3, min_lr=1e-6)
]

# 7. model training
print("\n=== Start training ===")
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=512,
                    validation_split=0.1,  # Divide 20% from the training set as validation set
                    callbacks=callbacks,
                    verbose=2)

# 8. Performance Visualisation
plt.figure(figsize=(12, 5))

# accurancy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('FCN (MNIST) Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.grid(True)

# loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.title('FCN (MNIST) Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


"""
第二部分： CNN的MNIST数据集测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 1. Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert to PyTorch tensor and add channel dimensions
x_train = torch.tensor(x_train.reshape(-1, 1, 28, 28), dtype=torch.float32) / 255.0
x_test = torch.tensor(x_test.reshape(-1, 1, 28, 28), dtype=torch.float32) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Creating datasets and data loaders
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024)

# 2. Modified CNN model (adapted to MNIST input size)
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            # Input channel changed to 1, kernel adjusted to 5x5 to fit 28x28 size
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(128, 256, kernel_size=4, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2),  # 7x7

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)  # Output changed to category 10
        )

    def forward(self, x):
        return self.features(x)

# Initialising the model
model = MNIST_CNN().to(device)
print(model)

# 3. Training configuration
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005,
                     steps_per_epoch=len(train_loader),
                     epochs=10)

def train_model():
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    model.train()
    for epoch in range(10):
        # Training phase
        epoch_train_loss = 0.0
        correct = 0
        total = 0

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

        # Calculation of training indicators
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100. * correct / total

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

        # Recording of indicators
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

    # Visualisation results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='training')
    plt.plot(val_accuracies, label='validation')
    plt.title('CNN (MINIST) Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='training')
    plt.plot(val_losses, label='validation')
    plt.title('CNN (MNIST) Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # final test
    model.eval()
    final_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            final_correct += predicted.eq(target).sum().item()

    final_acc = 100. * final_correct / len(test_dataset)
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")

# Start training
train_model()


"第三部分：FusionNet的MNIST测试"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa

import tensorflow as tf  # <--- Add this import
from tensorflow import keras
from tensorflow.keras import layers

# ========== GPU Configuration ==========
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for the first GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected - Using CPU")

# Rest of the original code follows...
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



# ========== Data Loading and Preprocessing ==========
# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

def preprocess_data(X_train, X_test, y_train, y_test):
    # Normalization
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN input
    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test.reshape(-1, 28, 28, 1)
    
    # Flatten for FCN input
    X_train_fcn = X_train.reshape(-1, 28*28)
    X_test_fcn = X_test.reshape(-1, 28*28)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train_cnn, X_test_cnn, X_train_fcn, X_test_fcn, y_train, y_test

X_train_cnn, X_test_cnn, X_train_fcn, X_test_fcn, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)

# Create validation set
X_train_cnn, X_val_cnn, X_train_fcn, X_val_fcn, y_train, y_val = train_test_split(
    X_train_cnn, X_train_fcn, y_train, test_size=0.1, random_state=42)

# ========== Enhanced FusionNet Architecture ==========
def build_fusionnet():
    # CNN Branch
    cnn_input = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(128, (4,4), activation='gelu', padding='same')(cnn_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (2,2), activation='gelu')(x)
    x = layers.GlobalMaxPooling2D()(x)
    
    # FCN Branch
    fcn_input = layers.Input(shape=(784,))
    y = layers.Dense(1024, activation='relu')(fcn_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(512, activation='relu')(y)  # Fixed connection
    y = layers.BatchNormalization()(y)
    
    # Feature Fusion
    combined = layers.concatenate([x, y])
    
    # Deep Processing
    z = layers.Dense(1024, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(512, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    
    # Output Layer
    outputs = layers.Dense(10, activation='softmax')(z)  # 10 classes for MNIST
    
    return keras.Model(inputs=[cnn_input, fcn_input], outputs=outputs)

# ========== Learning Rate Schedule ==========
def lr_schedule(epoch):
    return 1e-3 * (0.7 ** epoch)

# ========== Model Compilation ==========
model = build_fusionnet()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=tfa.losses.SigmoidFocalCrossEntropy(),
    metrics=['accuracy']
)

# ========== Model Training ==========
history = model.fit(
    [X_train_cnn, X_train_fcn], y_train,
    batch_size=256,
    epochs=10,
    validation_data=([X_val_cnn, X_val_fcn], y_val),
    callbacks=[LearningRateScheduler(lr_schedule)],
    verbose=1
)

# ========== Model Evaluation ==========
test_loss, test_acc = model.evaluate([X_test_cnn, X_test_fcn], y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# ========== Training Visualization ==========
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('FusionNet(MNIST) Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.title('FusionNet(MNIST) Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()