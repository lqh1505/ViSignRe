"""Normalize keypoint sequences for collection and training."""

import numpy as np

ZERO_THRESHOLD = 1e-6
MAX_GAP = 3


def any_hand_active(kp: np.ndarray) -> bool:
    return (
        np.linalg.norm(kp[0:63]) > ZERO_THRESHOLD
        or np.linalg.norm(kp[63:126]) > ZERO_THRESHOLD
    )


def fill_gaps(frames: np.ndarray, max_gap: int = MAX_GAP) -> np.ndarray:
    result = frames.copy()
    T = len(result)
    active = [any_hand_active(result[i]) for i in range(T)]

    active_indices = [i for i, a in enumerate(active) if a]
    if len(active_indices) < 2:
        return result

    first_active = active_indices[0]
    last_active = active_indices[-1]

    i = first_active
    while i <= last_active:
        if not active[i]:
            j = i
            while j <= last_active and not active[j]:
                j += 1

            gap_len = j - i
            if gap_len <= max_gap and j <= last_active:
                prev = result[i - 1]
                nxt = result[j]
                for k in range(gap_len):
                    alpha = (k + 1) / (gap_len + 1)
                    result[i + k] = (1 - alpha) * prev + alpha * nxt
            i = j
        else:
            i += 1

    return result


def trim_and_center_pad(frames: np.ndarray, seq_len: int) -> np.ndarray:
    kp_size = frames.shape[1]
    active_indices = [i for i in range(len(frames)) if any_hand_active(frames[i])]

    if not active_indices:
        return np.zeros((seq_len, kp_size), dtype=frames.dtype)

    first = active_indices[0]
    last = active_indices[-1]
    core = frames[first:last + 1]

    if len(core) >= seq_len:
        start = (len(core) - seq_len) // 2
        return core[start:start + seq_len].copy()

    pad_total = seq_len - len(core)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return np.concatenate([
        np.zeros((pad_left, kp_size), dtype=frames.dtype),
        core,
        np.zeros((pad_right, kp_size), dtype=frames.dtype),
    ], axis=0)


def normalize_sequence(frames, seq_len: int) -> np.ndarray:
    arr = np.array(frames, dtype=np.float32)
    arr = fill_gaps(arr)
    arr = trim_and_center_pad(arr, seq_len)
    return arr


if __name__ == '__main__':
    import os
    import sys

    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            data = np.load(path)
            norm = normalize_sequence(data, seq_len=data.shape[0])

            def active_str(arr):
                return "".join(
                    "#" if any_hand_active(arr[i]) else "."
                    for i in range(len(arr))
                )

            name = os.path.basename(path)
            n_before = sum(any_hand_active(data[i]) for i in range(len(data)))
            n_after = sum(any_hand_active(norm[i]) for i in range(len(norm)))
            print(f"\n{name}")
            print(f"  before: {active_str(data)} ({n_before} active)")
            print(f"  after:  {active_str(norm)} ({n_after} active)")
    else:
        print("Usage: python -m src.utils.sequence_utils sample_0.npy ...")
