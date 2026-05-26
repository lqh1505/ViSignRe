"""Plot index-finger trajectories per vocabulary class."""

import os
import sys
import glob

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'data/dataset_words'
ACTIONS = [
    '21', 'Gap', 'Ha_Noi', 'Hung', 'Song',
    'Ten', 'Toi', 'Tuoi', 'Vui', 'Xin_chao', 'Blank',
]
MAX_SAMPLES_PER_CLASS = 25
ZERO_THRESHOLD = 1e-6


def get_index_finger_trajectory(seq: np.ndarray):
    trajectories = []

    left_hand = seq[:, 0:63]
    if np.linalg.norm(left_hand) > ZERO_THRESHOLD:
        x = left_hand[:, 24]
        y = left_hand[:, 25]
        active_idx = np.where(np.abs(x) + np.abs(y) > ZERO_THRESHOLD)[0]
        if len(active_idx) > 0:
            trajectories.append((x[active_idx], y[active_idx]))

    right_hand = seq[:, 63:126]
    if np.linalg.norm(right_hand) > ZERO_THRESHOLD:
        x = right_hand[:, 24]
        y = right_hand[:, 25]
        active_idx = np.where(np.abs(x) + np.abs(y) > ZERO_THRESHOLD)[0]
        if len(active_idx) > 0:
            trajectories.append((x[active_idx], y[active_idx]))

    return trajectories


def main():
    print("[INFO] Loading data and plotting trajectories")

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    fig.suptitle(
        'Spatial Trajectories by Class (Index Finger)',
        fontsize=18, fontweight='bold', y=0.96,
    )
    axes = axes.flatten()

    for i, action in enumerate(ACTIONS):
        ax = axes[i]
        action_dir = os.path.join(DATA_PATH, action)

        if not os.path.exists(action_dir):
            ax.set_title(f"{action} (no data)")
            continue

        files = glob.glob(os.path.join(action_dir, "*.npy"))
        np.random.shuffle(files)
        files_to_plot = files[:MAX_SAMPLES_PER_CLASS]

        for file in files_to_plot:
            seq = np.load(file)
            for x, y in get_index_finger_trajectory(seq):
                ax.plot(x, y, alpha=0.4, linewidth=1.5)
                ax.plot(x[0], y[0], marker='o', markersize=4, color='lime')
                ax.plot(x[-1], y[-1], marker='o', markersize=4, color='red')

        ax.plot(0, 0, marker='X', markersize=10, color='black', markeredgewidth=2)
        ax.set_title(action, fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.6)
        ax.axvline(0, color='black', linewidth=0.6)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_yaxis()

    for j in range(len(ACTIONS), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = 'reports/quy_dao_khong_gian.png'
    os.makedirs('reports', exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"[OK] Saved {save_path}")
    plt.show()


if __name__ == '__main__':
    main()
