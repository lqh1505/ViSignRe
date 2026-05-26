"""Train the ViSignRe BiLSTM classifier."""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.sequence_utils import normalize_sequence, any_hand_active

actions = Config.ACTIONS
SEED = 42
DATA_PATH = 'data/dataset_words'
MODEL_SAVE_PATH = 'models/ViSignRe.keras'
REPORT_DIR = 'reports'

SEQUENCE_LENGTH = 45
KP_SIZE = 126
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 300
LABEL_SMOOTHING = 0.1
AUGMENT_NOISE_STD = 0.005
AUGMENT_SCALE_RANGE = (0.9, 1.1)
ZERO_THRESHOLD = 1e-6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(SEED)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


def load_dataset(data_path, actions, seq_len, kp_size):
    sequences, labels = [], []
    stats = {}

    for action in actions:
        action_dir = os.path.join(data_path, action)
        if not os.path.isdir(action_dir):
            print(f"[WARN] Missing directory: {action_dir}")
            continue

        files = sorted(f for f in os.listdir(action_dir) if f.endswith('.npy'))
        count = 0

        for fname in files:
            path = os.path.join(action_dir, fname)
            try:
                seq = np.load(path)
                if seq.ndim != 2 or seq.shape[1] != kp_size:
                    continue

                seq = normalize_sequence(seq, seq_len)
                n_active = sum(any_hand_active(seq[i]) for i in range(seq_len))
                min_active = 2 if action == 'Blank' else 5

                if n_active < min_active:
                    continue

                sequences.append(seq)
                labels.append(np.where(actions == action)[0][0])
                count += 1
            except Exception as e:
                print(f"[ERROR] {path}: {e}")

        stats[action] = count

    print("\n[INFO] Dataset distribution:")
    for action, cnt in stats.items():
        print(f"  {action:15s}: {cnt:4d}")

    return np.array(sequences, dtype=np.float32), np.array(labels)


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    aug = seq.copy()
    for i in range(len(aug)):
        if not any_hand_active(aug[i]):
            continue
        for start in (0, 63):
            slot = aug[i, start:start + 63]
            if np.linalg.norm(slot) > ZERO_THRESHOLD:
                slot = slot * np.random.uniform(*AUGMENT_SCALE_RANGE)
                slot = slot + np.random.normal(
                    0, AUGMENT_NOISE_STD, slot.shape
                ).astype(np.float32)
                aug[i, start:start + 63] = slot
    return aug


def augment_dataset(X: np.ndarray, y: np.ndarray, factor: int = 2) -> tuple:
    X_aug = np.concatenate([
        X,
        *[np.array([augment_sequence(s) for s in X]) for _ in range(factor - 1)],
    ])
    y_aug = np.tile(y, factor)
    return X_aug, y_aug


def main():
    print("\n" + "=" * 55)
    print(" ViSignRe Training")
    print("=" * 55)

    X_raw, y_raw = load_dataset(DATA_PATH, actions, SEQUENCE_LENGTH, KP_SIZE)
    print(f"\n[INFO] Valid samples: {len(X_raw)} | shape: {X_raw.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED,
    )

    X_train, y_train = augment_dataset(X_train, y_train, factor=2)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    y_train_cat = to_categorical(y_train, num_classes=len(actions))
    y_val_cat = to_categorical(y_val, num_classes=len(actions))
    y_test_cat = to_categorical(y_test, num_classes=len(actions))

    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train,
    )
    class_weight_dict = dict(enumerate(class_weights))

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, KP_SIZE)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax'),
    ], name='ViSignRe')

    model.summary()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['categorical_accuracy'],
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ModelCheckpoint(
            MODEL_SAVE_PATH, save_best_only=True, monitor='val_categorical_accuracy',
        ),
        ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n{'=' * 55}")
    print(f" Test accuracy: {test_acc * 100:.2f}%")
    print("=" * 55)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=actions))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrix.png'))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['categorical_accuracy'], label='Train')
    plt.plot(history.history['val_categorical_accuracy'], label='Val')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(REPORT_DIR, 'learning_curve.png'))


if __name__ == '__main__':
    main()
