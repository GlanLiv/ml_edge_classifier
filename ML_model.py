import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image

#pip install numpy opencv-python tensorflow scikit-learn matplotlib

# Configuration
IMG_SIZE = 64  # Try 128 if your edges have more detail
EPOCHS = 10
BATCH_SIZE = 32
NUM_FOLDS = 5
DATA_PATHS = {
    'open': 'class_open',
    'closed': 'class_closed'
}
MODEL_SAVE_PATH = 'best_edge_classifier_model.h5'

# Load images from folder
def load_images_from_folder(folder, label, img_size):
    images, labels = [], []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('L')  # Graustufen
                    img = img.resize((img_size, img_size))
                    img_np = np.array(img)
                    images.append(img_np)
                    labels.append(label)
            except Exception as e:
                print(f"⚠️ Failed to load {img_path}: {e}")
    return images, labels

# Plot training history
def plot_history(history, fold):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mse = history.history['mean_squared_error']
    val_mse = history.history['val_mean_squared_error']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(16, 5))

    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.title(f'Fold {fold} - Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title(f'Fold {fold} - Loss')
    plt.legend()

    # MSE
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, mse, label='Train MSE')
    plt.plot(epochs_range, val_mse, label='Val MSE')
    plt.title(f'Fold {fold} - MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Build the CNN model
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.MeanSquaredError()]
    )
    return model

# Load all images and labels
images_open, labels_open = load_images_from_folder(DATA_PATHS['open'], label=1, img_size=IMG_SIZE)
images_closed, labels_closed = load_images_from_folder(DATA_PATHS['closed'], label=0, img_size=IMG_SIZE)

X = np.array(images_open + images_closed, dtype=np.float32) / 255.0
y = np.array(labels_open + labels_closed)

# Reshape X to include channel dimension (grayscale: 1 channel)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# K-Fold Cross-Validation
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold = 1
accuracies = []
best_val_acc = 0.0
best_model = None

for train_idx, val_idx in kf.split(X):
    print(f"\n--- Fold {fold}/{NUM_FOLDS} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1))

    # Optional: Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate and store results
    val_loss, val_acc, val_mse = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}, MSE: {val_mse:.4f}")
    accuracies.append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        model.save(MODEL_SAVE_PATH)
        print(f"✅ Saved best model to {MODEL_SAVE_PATH}")

    # Visualize
    plot_history(history, fold)

    # Clean up
    K.clear_session()
    fold += 1

# Final results
print("\n=== Cross-Validation Summary ===")
for i, acc in enumerate(accuracies, 1):
    print(f"Fold {i}: {acc:.4f}")
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
